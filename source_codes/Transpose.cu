#include <iostream>
#include <stdio.h>
#include "Transpose.cuh"


__global__ void transposeKernel(float* dst, float* src, int nx, int ny)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int loc1 = i*nx + j;
	int loc2 = j*ny + i;
	dst[loc2] = src[loc1];
}

bool transposeCuda(float* const src_dst, int const nx, int const ny, int const cuda_dev_id)
{
	int const n_data = nx*ny;

	cudaError_t cudaStatus = cudaSuccess;

	// set cuda device;
	cudaStatus = cudaSetDevice(cuda_dev_id);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Unable to set CUDA device " << cuda_dev_id << ", " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Allocate GPU buffers for vectors
	float *_d_src = nullptr;
	float *_d_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_src, n_data * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_dst, n_data * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src, src_dst, n_data * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaDeviceReset();
		return false;
	}

	// determine thread size and block size
	int const block_dim = nx;
	int const grid_dim = ny;
	// Launch a kernel on the GPU with one thread for each element.
	transposeKernel << <grid_dim, block_dim >> >(_d_dst, _d_src, nx, ny);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform addition, " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaDeviceReset();
		return false;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to synchronized, " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(src_dst, _d_dst, n_data * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src);
	cudaFree(_d_dst);
	cudaDeviceReset();
	return true;
}


void transpose(float* const src_dst, int const nx, int const ny)
{// cpu
	int const num_byte = sizeof(float);
	float* tmp_arr = new float[nx*ny];
	std::memcpy(&tmp_arr[0], &src_dst[0], num_byte*nx*ny);
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			int loc1 = i*nx + j;
			int loc2 = j*ny + i;
			src_dst[loc2] = tmp_arr[loc1];
		}
	}
	delete[] tmp_arr;
}

