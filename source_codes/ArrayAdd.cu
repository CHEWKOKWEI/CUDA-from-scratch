#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "ArrayAdd.cuh"


__global__ void addKernel1D(float *dst, float *src1, float *src2, int n_data)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < n_data) { dst[i] = src1[i] + src2[i]; }
}

bool arrayAddCuda(float* const dst, float *const src1, float *const src2, int const n_data, int const cuda_dev_id)
{
	cudaError_t cudaStatus = cudaSuccess;

	// set cuda device;
	cudaStatus = cudaSetDevice(cuda_dev_id);
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Unable to set CUDA device " << cuda_dev_id << ", " << cudaGetErrorString(cudaStatus) <<"." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Allocate GPU buffers for three vectors
	float *_d_src1 = nullptr;
	float *_d_src2 = nullptr;
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_dst, n_data * sizeof(float));
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_src1, n_data * sizeof(float));
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_src2, n_data * sizeof(float));
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src1, src1, n_data * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(_d_src2, src2, n_data * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// determine thread size and block size
	int const max_thread_size = 1024;
	int block_dim = 0;
	int grid_dim = 0;
	if (n_data <= max_thread_size)
	{
		block_dim = n_data;
		grid_dim = 1;
	}
	else
	{
		block_dim = max_thread_size;
		grid_dim = n_data / max_thread_size + (int)((n_data%max_thread_size) > 0);
	}
	// Launch a kernel on the GPU with one thread for each element.
	addKernel1D << <grid_dim, block_dim >> >(_d_dst, _d_src1, _d_src2, n_data);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Failed to perform addition, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(dst, _d_dst, n_data * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src1);
	cudaFree(_d_src2);
	cudaFree(_d_dst);
	cudaDeviceReset();
	return true;
}


void arrayAdd(float* const dst, float* const src1, float* const src2, int const n_data)
{// cpu
	for (int i = 0; i < n_data; i++) { dst[i] = src1[i] + src2[i]; }
}
