#include <iostream>
#include <stdio.h>
#include "Transpose.cuh"


__global__ void flipHorizontalKernel(float* src_dst, int nx, int ny)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int loc1 = i*nx + j;
	int loc2 = i*nx + (nx - 1 - j);
	float tmp = src_dst[loc2];
	__syncthreads();
	src_dst[loc1] = tmp;
}

__global__ void flipVerticalKernel(float* src_dst, int nx, int ny)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int loc1 = i*nx + j;
	int loc2 = (ny - 1 - i)*nx + j;
	float tmp = src_dst[loc1];
	__syncthreads();
	src_dst[loc1] = src_dst[loc2];
}


bool flipHorizontalCuda(float* const src_dst, int const nx, int const ny, int const cuda_dev_id)
{
	int const n_data = nx*ny;

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
	float *_d_src_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_src_dst, n_data * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src_dst, src_dst, n_data * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// determine thread size and block size
	int const block_dim = nx;
	int const grid_dim = ny;
	// Launch a kernel on the GPU with one thread for each element.
	flipHorizontalKernel << <grid_dim, block_dim >> >(_d_src_dst, nx, ny);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform fliping, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(src_dst, _d_src_dst, n_data * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src_dst);
	cudaDeviceReset();
	return true;
}

bool flipVerticalCuda(float* const src_dst, int const nx, int const ny, int const cuda_dev_id)
{
	int const n_data = nx*ny;

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
	float *_d_src_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_src_dst, n_data * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src_dst, src_dst, n_data * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// determine thread size and block size
	int const block_dim = ny;
	int const grid_dim = nx;
	// Launch a kernel on the GPU with one thread for each element.
	flipVerticalKernel << <grid_dim, block_dim >> >(_d_src_dst, nx, ny);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform fliping, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(src_dst, _d_src_dst, n_data * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src_dst);
	cudaDeviceReset();
	return true;
}


void flipHorizontal(float* const src_dst, int const nx, int const ny)
{// cpu
	float tmp = 0;
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx / 2; j++)
		{
			int loc1 = i*nx + j;
			int loc2 = i*nx + (nx - 1 - j);
			tmp = src_dst[loc1];
			src_dst[loc1] = src_dst[loc2];
			src_dst[loc2] = tmp;
		}
	}
}

void flipVertical(float* const src_dst, int const nx, int const ny)
{// cpu
	float tmp = 0;
	for (int i = 0; i < ny / 2; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			int loc1 = i*nx + j;
			int loc2 = (ny - 1 - i)*nx + j;
			tmp = src_dst[loc1];
			src_dst[loc1] = src_dst[loc2];
			src_dst[loc2] = tmp;
		}
	}
}


