#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "Convolution.cuh"

__global__ void convolve2dKernel(float *dst, float *src, float *kernel, int src_nx, int src_ny, int kern_nx, int kern_ny)
{
	int dst_nx = src_nx - kern_nx + 1;
	int dst_ny = src_ny - kern_ny + 1;
	int i = blockIdx.x;
	int j = threadIdx.x;
	int loc_dst = i*dst_nx + j;
	float sum = 0;
	for (int p = 0; p < kern_ny; p++)
	{
		for (int q = 0; q < kern_nx; q++)
		{
			int loc_src = (i + p)*src_nx + (j + q);
			int loc_kern = p*kern_nx + q;
			sum += src[loc_src] * kernel[loc_kern];
		}
	}
	dst[loc_dst] = sum;
}

__global__ void convolve2dKernel2(float *dst, float *src, float *kernel, int src_nx, int src_ny, int kern_nx, int kern_ny)
{
	int dst_nx = src_nx - kern_nx + 1;
	int dst_ny = src_ny - kern_ny + 1;
	int i = blockIdx.x;
	int j = blockIdx.y;
	int p = threadIdx.x;
	int q = threadIdx.y;
	int loc_dst = i*dst_nx + j;
	//__shared__ float window[512];
	__shared__ float product[512];
	int loc_src = (i + p)*src_nx + (j + q);
	int loc_kern = p*kern_nx + q;
	product[loc_kern] = src[loc_src] * kernel[loc_kern];
	__syncthreads();
	float sum = 0;
	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		for (int m = 0; m < kern_ny*kern_nx; m++)
		{
			sum += product[m];
		}
		dst[loc_dst] = sum;
	}
}

__global__ void convolveFull2dKernel(float *dst, float *src, float *kernel, int src_nx, int src_ny, int kern_nx, int kern_ny)
{
	int dst_nx = src_nx;
	int dst_ny = src_ny;
	int i = blockIdx.x;
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int loc_dst = i*dst_nx + j;
	if (j >= src_nx) { return; }
	float sum = 0;
	for (int p = 0; p < kern_ny; p++)
	{
		for (int q = 0; q < kern_nx; q++)
		{
			int loc_src_y = (i + p - p / 2);
			int loc_src_x = (j + q - q / 2);
			loc_src_y = loc_src_y * (int)(loc_src_y > 0);
			loc_src_y = loc_src_y - (loc_src_y - src_ny + 1) * (int)(loc_src_y >= src_ny);
			loc_src_x = loc_src_x * (int)(loc_src_x > 0);
			loc_src_x = loc_src_x - (loc_src_x - src_nx + 1) * (int)(loc_src_x >= src_nx);
			int loc_src = loc_src_y * src_nx + loc_src_x;
			loc_src = loc_src * (int)(loc_src > 0);
			int loc_kern = p*kern_nx + q;
			sum += src[loc_src] * kernel[loc_kern];
		}
	}
	dst[loc_dst] = sum;
}


bool convolve2dCuda(float* const dst, float *const src, float *const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny, int const cuda_dev_id)
{
	int conv_nx = src_nx - kern_nx + 1;
	int conv_ny = src_ny - kern_ny + 1;

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
	float *_d_kern = nullptr;
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_dst, conv_ny* conv_nx * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_src, src_nx * src_ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_kern, kern_nx * kern_ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src, src, src_nx * src_ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(_d_kern, kernel, kern_nx * kern_ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// determine thread size and block size
	int block_dim = conv_nx;
	int grid_dim = conv_ny;
	// Launch a kernel on the GPU with one thread for each element.
	convolve2dKernel << <grid_dim, block_dim >> >(_d_dst, _d_src, _d_kern, src_nx, src_ny, kern_nx, kern_ny);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform convolution, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(dst, _d_dst, conv_ny* conv_nx * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src);
	cudaFree(_d_kern);
	cudaFree(_d_dst);
	cudaDeviceReset();
	return true;
}

bool convolve2dCuda2(float* const dst, float *const src, float *const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny, int const cuda_dev_id)
{
	int conv_nx = src_nx - kern_nx + 1;
	int conv_ny = src_ny - kern_ny + 1;

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
	float *_d_kern = nullptr;
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_dst, conv_ny* conv_nx * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_src, src_nx * src_ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_kern, kern_nx * kern_ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src, src, src_nx * src_ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(_d_kern, kernel, kern_nx * kern_ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// determine thread size and block size
	dim3 block_dim(kern_ny, kern_nx);
	dim3 grid_dim(conv_ny, conv_nx);
	// Launch a kernel on the GPU with one thread for each element.
	convolve2dKernel2 << <grid_dim, block_dim >> >(_d_dst, _d_src, _d_kern, src_nx, src_ny, kern_nx, kern_ny);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform convolution, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(dst, _d_dst, conv_ny* conv_nx * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src);
	cudaFree(_d_kern);
	cudaFree(_d_dst);
	cudaDeviceReset();
	return true;
}

bool convolveFull2dCuda(float* const dst, float *const src, float *const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny, int block_size, int const cuda_dev_id)
{
	int conv_nx = src_nx;
	int conv_ny = src_ny;
	if ((kern_nx % 2 == 0)||(kern_nx<0))
	{
		std::cout << "Kernel size must be positive odd number." << std::endl;
		return false;
	}
	if ((kern_ny % 2 == 0) || (kern_ny<0))
	{
		std::cout << "Kernel size must be positive odd number." << std::endl;
		return false;
	}

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
	float *_d_kern = nullptr;
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_dst, conv_ny* conv_nx * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_src, src_nx * src_ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_kern, kern_nx * kern_ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src, src, src_nx * src_ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(_d_kern, kernel, kern_nx * kern_ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// determine thread size and block size
	int block_dim = block_size;
	dim3 grid_dim(conv_ny, (conv_nx / block_size + (int)((conv_nx % block_size) >0 )));
	// Launch a kernel on the GPU with one thread for each element.
	convolveFull2dKernel << <grid_dim, block_dim >> >(_d_dst, _d_src, _d_kern, src_nx, src_ny, kern_nx, kern_ny);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform convolution, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(dst, _d_dst, conv_ny* conv_nx * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src);
	cudaFree(_d_kern);
	cudaFree(_d_dst);
	cudaDeviceReset();
	return true;
}


void convolve2d(float* const dst, float* const src, float* const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny)
{// cpu
	int const dst_nx = src_nx - kern_nx + 1;
	int const dst_ny = src_ny - kern_ny + 1;
	for (int i = 0; i < dst_ny; i++)
	{
		for (int j = 0; j < dst_nx; j++)
		{
			int loc_dst = i*dst_nx + j;
			float sum = 0;
			for (int p = 0; p < kern_ny; p++)
			{
				for (int q = 0; q < kern_nx; q++)
				{
					int loc_src = (i + p)*src_nx + (j + q);
					int loc_kern = p*kern_nx + q;
					sum += src[loc_src] * kernel[loc_kern];
				}
			}
			dst[loc_dst] = sum;
		}
	}
}

