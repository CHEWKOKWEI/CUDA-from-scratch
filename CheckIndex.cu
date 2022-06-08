#include <iostream>
#include "CheckIndex.cuh"

__global__ void checkIndexKernel1D_1D(int *thread_id, int *block_id, int *global_id, int n_data)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < n_data) 
	{
		thread_id[i] = threadIdx.x;
		block_id[i] = blockIdx.x;
		global_id[i] = i;
	}
}

__global__ void checkIndexKernel2D_2D(int* thread_id_x, int* thread_id_y, int* block_id_x, int* block_id_y, int*  global_id, int n_data)
{
	int i = threadIdx.x + 
		threadIdx.y * blockDim.x +
		blockIdx.x * (blockDim.x*blockDim.y) +
		blockIdx.y * (gridDim.x*blockDim.x*blockDim.y);
	if (i < n_data)
	{
		thread_id_x[i] = threadIdx.x;
		thread_id_y[i] = threadIdx.y;
		block_id_x[i] = blockIdx.x;
		block_id_y[i] = blockIdx.y;
		global_id[i] = i;
	}
}

bool checkIndex1Dgrid_1Dblock()
{
	std::cout << "Cuda Indexing - 1D Grid of 1D Block" << std::endl;
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Unable to set CUDA device " << 0 << ", " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	int const n_thread = 5;
	int const n_block = 4;
	int const n_data = n_block*n_thread;
	int* thread_id = new int[n_data];
	int* block_id = new int[n_data];
	int* global_id = new int[n_data];

	int* _d_thread_id = 0;
	int* _d_block_id = 0;
	int* _d_global_id = 0;
	cudaStatus = cudaMalloc((void**)&_d_thread_id, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_block_id, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_global_id, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	checkIndexKernel1D_1D << <n_block, n_thread >> >(_d_thread_id, _d_block_id, _d_global_id, n_data);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform check index, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to synchronized, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	cudaStatus = cudaMemcpy(thread_id, _d_thread_id, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(block_id, _d_block_id, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(global_id, _d_global_id, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_thread_id);
	cudaFree(_d_block_id);
	cudaFree(_d_global_id);
	cudaDeviceReset();
	std::cout << "Number of blocks(grid size)=" << n_block << ", Number of threads(block size)=" << n_thread << std::endl;
	std::cout << " Thread Id\tBlock Id\tGlobal Id" << std::endl;
	for (int i = 0; i < n_data; i++)
	{
		std::cout << "   "  << thread_id[i] << "\t\t  " << block_id[i] << "\t\t  " << global_id[i] << std::endl;
	}

	delete[] thread_id, block_id, global_id;
	return true;
}


bool checkIndex2Dgrid_2Dblock()
{
	std::cout << "Cuda Indexing - 2D Grid of 2D Block" << std::endl;
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Unable to set CUDA device " << 0 << ", " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	int n_thread_x = 2;
	int n_thread_y = 3;
	int n_block_x = 4;
	int n_block_y = 5;
	int n_data = n_thread_x*n_thread_y*n_block_x*n_block_y;
	int* thread_id_x = new int[n_data];
	int* thread_id_y = new int[n_data];
	int* block_id_x = new int[n_data];
	int* block_id_y = new int[n_data];
	int* global_id = new int[n_data];

	int* _d_thread_id_x = 0;
	int* _d_thread_id_y = 0;
	int* _d_block_id_x = 0;
	int* _d_block_id_y = 0;
	int* _d_global_id = 0;
	cudaStatus = cudaMalloc((void**)&_d_thread_id_x, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_thread_id_y, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_block_id_x, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_block_id_y, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMalloc((void**)&_d_global_id, n_data * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	dim3 grid_dim(n_block_x, n_block_y);
	dim3 block_dim(n_thread_x, n_thread_y);
	checkIndexKernel2D_2D << <grid_dim, block_dim >> >(_d_thread_id_x, _d_thread_id_y, _d_block_id_x, _d_block_id_y, _d_global_id, n_data);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Failed to perform check index, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "  Failed to synchronized, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	cudaStatus = cudaMemcpy(thread_id_x, _d_thread_id_x, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(thread_id_y, _d_thread_id_y, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(block_id_x, _d_block_id_x, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(block_id_y, _d_block_id_y, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaStatus = cudaMemcpy(global_id, _d_global_id, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_thread_id_x);
	cudaFree(_d_thread_id_y);
	cudaFree(_d_block_id_x);
	cudaFree(_d_block_id_y);
	cudaFree(_d_global_id);
	cudaDeviceReset();
	std::cout << "Number of blocks(grid size)=" << grid_dim.x<<"x"<< grid_dim.y << ", Number of threads(block size)=" << block_dim.x<<"x" << block_dim.y<< std::endl;
	std::cout << " Thread Id(x)\tThread Id(y)\tBlock Id(x)\tBlock Id(y)\tGlobal Id" << std::endl;
	for (int i = 0; i < n_data; i++)
	{
		std::cout << "   " << thread_id_x[i] << "\t\t  " << thread_id_y[i] << "\t\t  " << block_id_x[i] << "\t\t  " << block_id_y[i] << "\t\t  " <<global_id[i] << std::endl;
	}
	delete[] thread_id_x, thread_id_y, block_id_x, block_id_y, global_id;
	return true;
}
