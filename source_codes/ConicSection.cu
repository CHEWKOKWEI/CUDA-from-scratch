#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <texture_types.h>
#include <curand.h>
#include "ConicSection.cuh"

__device__ float trilinearInterpolate(float *src, int src_nx, int src_ny, int src_nz, float x, float y, float z)
{
	float delta = 1.0e-3;
	float x1 = floorf(x);
	float x2 = ceilf(x);
	float y1 = floorf(y);
	float y2 = ceilf(y);
	float z1 = floorf(z);
	float z2 = ceilf(z);
	int x1_idx = (int)x1;
	int x2_idx = (int)x2;
	int y1_idx = (int)y1;
	int y2_idx = (int)y2;
	int z1_idx = (int)z1;
	int z2_idx = (int)z2;
	float f111 = src[x1_idx + y1_idx*src_nx + z1_idx*src_nx*src_ny];
	float f112 = src[x2_idx + y1_idx*src_nx + z1_idx*src_nx*src_ny];
	float f121 = src[x1_idx + y2_idx*src_nx + z1_idx*src_nx*src_ny];
	float f122 = src[x2_idx + y2_idx*src_nx + z1_idx*src_nx*src_ny];
	float f211 = src[x1_idx + y1_idx*src_nx + z2_idx*src_nx*src_ny];
	float f212 = src[x2_idx + y1_idx*src_nx + z2_idx*src_nx*src_ny];
	float f221 = src[x1_idx + y2_idx*src_nx + z2_idx*src_nx*src_ny];
	float f222 = src[x2_idx + y2_idx*src_nx + z2_idx*src_nx*src_ny];
	float f11x = f111 + (f112 - f111)*(x - x1) / max(x2 - x1, delta);
	float f12x = f121 + (f122 - f121)*(x - x1) / max(x2 - x1, delta);
	float f21x = f211 + (f212 - f211)*(x - x1) / max(x2 - x1, delta);
	float f22x = f221 + (f222 - f221)*(x - x1) / max(x2 - x1, delta);
	float f1yx = f11x + (f12x - f11x)*(y - y1) / max(y2 - y1, delta);
	float f2yx = f21x + (f22x - f21x)*(y - y1) / max(y2 - y1, delta);
	float fzyx = f1yx + (f2yx - f1yx)*(z - z1) / max(z2 - z1, delta);
	return fzyx;
}

__global__ void slicingKernel(float *dst, int dst_nx, int dst_ny,
	float *src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float  *angle_s)
{
	float factor = 1.0;
	int i = threadIdx.x;
	int j = blockIdx.x;
	int k = blockIdx.y;
	int loc_dst = i + j*dst_nx +k*dst_nx*dst_ny;
	float src_y = y0 + j / factor - (float)(dst_ny) / 2.0 / factor;
	float src_x0 = i - (float)(dst_nx) / 2.0;
	float src_z0 = 0;
	float src_x = (src_x0)*std::cos(angle_s[k]) / factor - (src_z0)*std::sin(angle_s[k]) / factor + x0;
	float src_z = (src_x0)*std::sin(angle_s[k]) / factor + (src_z0)*std::cos(angle_s[k]) / factor + z0;
	/*int idx_x = (int)(src_x);
	int idx_y = (int)(src_y);
	int idx_z = (int)(src_z);
	int loc_src = idx_x + idx_y*src_nx + idx_z * src_nx*src_ny; //i + j*src_nx + 256 * src_nx*src_ny;
	dst[loc_dst] = src[loc_src];*/
	float a = trilinearInterpolate(src, src_nx, src_ny, src_nz, src_x, src_y, src_z);
	dst[loc_dst] = a;
}

__global__ void slicingWithTextureKernel(float *dst, int dst_nx, int dst_ny,
	cudaTextureObject_t tex_src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float *angle_s)
{
	float factor = 1.0;
	int i = threadIdx.x;
	int j = blockIdx.x;
	int k = blockIdx.y;
	int loc_dst = i + j*dst_nx +k*dst_nx*dst_ny;
	float src_y = y0 + j / factor - (float)(dst_ny) / 2.0 / factor;
	float src_x0 = i - (float)(dst_nx) / 2.0;
	float src_z0 = 0;
	float src_x = (src_x0)*std::cos(angle_s[k]) / factor - (src_z0)*std::sin(angle_s[k]) / factor + x0;
	float src_z = (src_x0)*std::sin(angle_s[k]) / factor + (src_z0)*std::cos(angle_s[k]) / factor + z0;
	float src_xn = src_x;// / (float)(src_nx);
	float src_yn = src_y;// / (float)(src_ny);
	float src_zn = src_z;// / (float)(src_nz);
	dst[loc_dst] = tex3D<float>(tex_src, src_xn, src_yn, src_zn);;
}


bool slicing(float *dst, int n_angle, int dst_nx, int dst_ny,
	float* const src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float* angle_s, int const cuda_dev_id)
{
	float n_voxel = src_nx*src_ny*src_nz;
	cudaError_t cudaStatus = cudaSuccess;

	// set cuda device;
	cudaStatus = cudaSetDevice(cuda_dev_id);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Unable to set CUDA device " << cuda_dev_id << ", " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Allocate GPU buffers for vectors
	float *_d_src = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_src, n_voxel * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
	std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
	cudaDeviceReset();
	return false;
	}
	float *_d_angle = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_angle, n_angle * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_dst, dst_nx* dst_ny * n_angle * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src, src, n_voxel * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
	std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
	cudaDeviceReset();
	return false;
	}
	cudaStatus = cudaMemcpy(_d_angle, angle_s, n_angle * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// launch kernel
	int block_dim = dst_nx;
	dim3 grid_dim(dst_ny, n_angle);
	slicingKernel << <grid_dim, block_dim >> >(_d_dst, dst_nx, dst_ny,
		_d_src, src_nx, src_ny, src_nz, x0, y0, z0, _d_angle);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform slicing, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to synchronized, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(dst, _d_dst, dst_nx* dst_ny * n_angle * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src);
	cudaFree(_d_angle);
	cudaFree(_d_dst);
	cudaDeviceReset();
	return true;
}

bool slicingWithTexture(float *dst, int n_angle, int dst_nx, int dst_ny,
	float* const src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float* angle_s, int const cuda_dev_id)

{
	float n_voxel = src_nx*src_ny*src_nz;
	cudaError_t cudaStatus = cudaSuccess;

	// set cuda device;
	cudaStatus = cudaSetDevice(cuda_dev_id);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Unable to set CUDA device " << cuda_dev_id << ", " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Allocate GPU buffers for vectors
	float *_d_angle = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_angle, n_angle * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_dst, dst_nx* dst_ny * n_angle * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_angle, angle_s, n_angle * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// cuda array creation
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *_d_cu_arr;
	cudaStatus = cudaMalloc3DArray(&_d_cu_arr, &channelDesc, make_cudaExtent(src_nx * sizeof(float), src_ny, src_nz), 0);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA 3D memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// copy array
	cudaMemcpy3DParms copy_params = { 0 };
	copy_params.srcPtr = make_cudaPitchedPtr(src, src_nx * sizeof(float), src_ny, src_nz);
	copy_params.dstArray = _d_cu_arr;
	copy_params.extent = make_cudaExtent(src_nx * 1, src_ny, src_nz);
	copy_params.kind = cudaMemcpyHostToDevice;
	cudaStatus = cudaMemcpy3D(&copy_params);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from jost to CUDA array, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// bind to texture
	cudaTextureObject_t _tex_src;
	cudaResourceDesc    tex_res;
	memset(&tex_res, 0, sizeof(cudaResourceDesc));
	tex_res.resType = cudaResourceTypeArray;
	tex_res.res.array.array = _d_cu_arr;
	cudaTextureDesc     tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));
	tex_desc.normalizedCoords = false;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.addressMode[0] = cudaAddressModeClamp; 
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeClamp;
	tex_desc.readMode = cudaReadModeElementType;
	cudaStatus = cudaCreateTextureObject(&_tex_src, &tex_res, &tex_desc, NULL);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to bind CUDA array to texture, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// launch kernel
	int block_dim = dst_nx;
	dim3 grid_dim(dst_ny,n_angle);
	slicingWithTextureKernel << <grid_dim, block_dim >> >(_d_dst, dst_nx, dst_ny,
		_tex_src, src_nx, src_ny, src_nz, x0, y0, z0, _d_angle);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform slicing, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to synchronized, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(dst, _d_dst, dst_nx* dst_ny * n_angle * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	//cudaFree(_d_src);
	cudaFree(_d_angle);
	cudaFree(_d_dst);
	cudaFreeArray(_d_cu_arr);
	cudaDeviceReset();
	return true;
}


bool slicingWithTextureAndStream(float *dst, int n_angle, int dst_nx, int dst_ny,
	float* const src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float* angle_s, int const cuda_dev_id)

{
	float n_voxel = src_nx*src_ny*src_nz;
	cudaError_t cudaStatus = cudaSuccess;

	// set cuda device;
	cudaStatus = cudaSetDevice(cuda_dev_id);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Unable to set CUDA device " << cuda_dev_id << ", " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Allocate GPU buffers for vectors
	float *_d_angle = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_angle, n_angle * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_dst, dst_nx* dst_ny * n_angle * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_angle, angle_s, n_angle * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// cuda array creation
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *_d_cu_arr;
	cudaStatus = cudaMalloc3DArray(&_d_cu_arr, &channelDesc, make_cudaExtent(src_nx * sizeof(float), src_ny, src_nz), 0);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA 3D memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// copy array
	cudaMemcpy3DParms copy_params = { 0 };
	copy_params.srcPtr = make_cudaPitchedPtr(src, src_nx * sizeof(float), src_ny, src_nz);
	copy_params.dstArray = _d_cu_arr;
	copy_params.extent = make_cudaExtent(src_nx * 1, src_ny, src_nz);
	copy_params.kind = cudaMemcpyHostToDevice;
	cudaStatus = cudaMemcpy3D(&copy_params);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to CUDA array, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// bind to texture
	cudaTextureObject_t _tex_src;
	cudaResourceDesc    tex_res;
	memset(&tex_res, 0, sizeof(cudaResourceDesc));
	tex_res.resType = cudaResourceTypeArray;
	tex_res.res.array.array = _d_cu_arr;
	cudaTextureDesc     tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));
	tex_desc.normalizedCoords = false;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.addressMode[0] = cudaAddressModeClamp;   // clamp
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeClamp;
	tex_desc.readMode = cudaReadModeElementType;
	cudaStatus = cudaCreateTextureObject(&_tex_src, &tex_res, &tex_desc, NULL);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to bind CUDA array to texture, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// declare stream
	const int n_stream = 4;
	cudaStream_t stream_s[n_stream];
	int n_set_per_stream = n_angle / n_stream;
	int n_data_per_stream = dst_nx*dst_ny*n_set_per_stream;
	int block_dim = dst_nx;
	dim3 grid_dim(dst_ny, n_set_per_stream);
	for (int i = 0; i < n_stream; i++)
	{
		// create stream
		cudaStreamCreate(&stream_s[i]);

		// perform calculation
		slicingWithTextureKernel << <grid_dim, block_dim, 0, stream_s[i]>> >(&_d_dst[n_data_per_stream*i], dst_nx, dst_ny,
			_tex_src, src_nx, src_ny, src_nz, x0, y0, z0, &_d_angle[i*n_set_per_stream]);

		// copy results async
		cudaMemcpyAsync(&dst[n_data_per_stream*i], &_d_dst[n_data_per_stream*i],
			n_data_per_stream * sizeof(float), cudaMemcpyDeviceToHost, stream_s[i]);
	}

	for (int i = 0; i < n_stream; ++i)
	{
		cudaStreamDestroy(stream_s[i]);
	}

	cudaFree(_d_angle);
	cudaFree(_d_dst);
	cudaFreeArray(_d_cu_arr);
	cudaDeviceReset();
	return true;
}



