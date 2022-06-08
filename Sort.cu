#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "Sort.cuh"

__device__ __host__ void selectionSort(float* arr, int start_idx, int stride, int n_data)
{
	int idx_i, idx_j, idx_min;
	float tmp_min;
	for (int i = 0; i < n_data - 1; i++)
	{
		idx_i = start_idx + (i*stride);
		idx_min = idx_i;
		tmp_min = arr[idx_i];
		for (int j = i + 1; j < n_data; j++)
		{
			idx_j = start_idx + (j*stride);
			if (arr[idx_j] < tmp_min) { idx_min = idx_j; tmp_min = arr[idx_j]; }
		}
		arr[idx_min] = arr[idx_i];
		arr[idx_i] = tmp_min;
	}
}

__device__ __host__ void insertionSort(float* arr, int start_idx, int stride, int n_data)
{
	for (int i = 1; i < n_data; i++)
	{
		for (int j = i; j > 0; j--)
		{
			float fore = arr[start_idx + (j - 1) * stride];
			float back = arr[start_idx + j * stride];
			if (back < fore)
			{
				arr[start_idx + j * stride] = fore;
				arr[start_idx + (j - 1) * stride] = back;
			}
			else { break; }
		}
	}
}

__device__ __host__ void buildMaxHeap(float* arr, int start_idx, int stride, int n_data)
{
	for (int i = 1; i < n_data; i++)
	{
		if (arr[start_idx + i*stride] > arr[start_idx + ((i - 1) / 2)*stride])
		{
			int j = i;
			while (arr[start_idx + j*stride] > arr[start_idx + ((j - 1) / 2)*stride])
			{
				float tmp = arr[start_idx + j*stride];
				arr[start_idx + j*stride] = arr[start_idx + ((j - 1) / 2)*stride];
				arr[start_idx + ((j - 1) / 2)*stride] = tmp;
				j = (j - 1) / 2;
			}
		}
	}
}

__device__ __host__ void heapSort(float* arr, int start_idx, int stride, int n_data)
{
	buildMaxHeap(arr, start_idx, stride, n_data);
	for (int i = n_data - 1; i > 0; i--)
	{
		float tmp = arr[start_idx];
		arr[start_idx] = arr[start_idx + i*stride];
		arr[start_idx + i*stride] = tmp;
		int j = 0, index = 0;
		while (index < i)
		{
			index = (2 * j + 1);
			if (index >= i) { continue; }
			if (arr[start_idx + index*stride] < arr[start_idx + (index + 1)*stride] && index < (i - 1)) { index++; }
			if (arr[start_idx + j*stride] < arr[start_idx + index*stride] && index < i)
			{
				float tmp = arr[start_idx + j*stride];
				arr[start_idx + j*stride] = arr[start_idx + index*stride];
				arr[start_idx + index*stride] = tmp;
			}
			j = index;
		}
	}
}

__device__ __host__ void partition(float* arr, int start_idx, int stride, int low, int high, int& partition_index)
{
	float pivot = arr[start_idx + high*stride];
	int i = (low - 1);
	float tmp = 0;
	for (int j = low; j <= high - 1; j++)
	{
		if (arr[start_idx + j*stride] <= pivot)
		{
			i++;
			tmp = arr[start_idx + i*stride];
			arr[start_idx + i*stride] = arr[start_idx + j*stride];
			arr[start_idx + j*stride] = tmp;
		}
	}
	tmp = arr[start_idx + (i + 1)*stride];
	arr[start_idx + (i + 1)*stride] = arr[start_idx + high*stride];
	arr[start_idx + high*stride] = tmp;
	partition_index = i + 1;
}

__device__ __host__ void quickSort(float* arr, int start_idx, int stride, int low, int high)
{
	float stack[1024] = { 0 };
	int top = -1;
	stack[++top] = low;
	stack[++top] = high;
	while (top >= 0)
	{
		high = stack[top--];
		low = stack[top--];
		int partition_index = 0;
		partition(arr, start_idx, stride, low, high, partition_index);
		if (partition_index - 1 > low)
		{
			stack[++top] = low;
			stack[++top] = partition_index - 1;
		}
		if (partition_index + 1 < high)
		{
			stack[++top] = partition_index + 1;
			stack[++top] = high;
		}
	}
}

__device__ __host__ void bitonicSwap(float &src1, float &src2, int reverse)
{
	float tmp1 = src1;
	float tmp2 = src2;
	//int smaller = (int)(tmp2 < tmp1);
	if (reverse == 1)
	{
		if (tmp2 < tmp1) { src1 = tmp1; src2 = tmp2; }
		else { src1 = tmp2; src2 = tmp1; }
	}
	else
	{
		if (tmp2 < tmp1) { src1 = tmp2; src2 = tmp1; }
		else { src1 = tmp1; src2 = tmp2; }
	}
}

__device__ __host__ void bitonicSortSimple(float* arr, int start_idx, int stride, int n_data)
{// no virtual padding, use only for array size 2^N
	const int n_round = ceil(log2f(n_data));
	const int n_virtual = powf(2, n_round);
	const int n_missing0 = n_virtual - n_data;
	int nbswap0 = 1;
	for (int i = 0; i < n_round; i++)
	{
		int nbswap = nbswap0;
		for (int j = i; j >= 0; j--)
		{
			int n_set = n_virtual / nbswap / 2;
			for (int k = 0; k < n_set; k++)
			{
				for (int m = 0; m < nbswap; m++)
				{
					int loc1 = k*nbswap * 2 + m;
					int loc2 = loc1 + nbswap;
					int reverse = (loc1 / nbswap0 / 2) % 2;
					//std::cout << "rev: " << reverse << std::endl;
					bitonicSwap(arr[start_idx + loc1*stride], arr[start_idx + loc2*stride], reverse);
				}
			}
			nbswap /= 2;
		}
		nbswap0 *= 2;
		//if (i==10)break;
	}
}

__device__ __host__ void bitonicSortVirtualPad(float* arr, int start_idx, int stride, int n_data)
{
	const int n_round = ceil(log2f(n_data));
	const int n_virtual = powf(2, n_round);
	const int n_missing = n_virtual - n_data;
	float very_big = 3e38;
	float* v_pad = 0;
	//float v_pad[4096];
	if (n_missing != 0)
	{
		v_pad = new float[n_missing];
		for (int i = 0; i < n_missing; i++) { v_pad[i] = very_big; }
	}
	int nbswap0 = 1;
	for (int i = 0; i < n_round; i++)
	{
		int nbswap = nbswap0;
		for (int j = i; j >= 0; j--)
		{
			int n_set = n_virtual / nbswap / 2;
			for (int k = 0; k < n_set; k++)
			{
				for (int m = 0; m < nbswap; m++)
				{
					int loc1 = k*nbswap * 2 + m;
					int loc2 = loc1 + nbswap;
					int reverse = (loc1 / nbswap0 / 2) % 2;
					if (loc1 >= n_data)
					{
						bitonicSwap(v_pad[loc1 - n_data], v_pad[loc2 - n_data], reverse);
					}
					else if (loc2 >= n_data)
					{
						bitonicSwap(arr[start_idx + loc1*stride], v_pad[loc2-n_data], reverse);
					}
					else
					{
						bitonicSwap(arr[start_idx + loc1*stride], arr[start_idx + loc2*stride], reverse);
					}
				}
			}
			nbswap /= 2;
		}
		nbswap0 *= 2;
		//if (i==10)break;
	}
	if (n_missing != 0)
	{
		delete[] v_pad;
	}
	return;/**/
}


__global__ void sortKernel(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride)
{// sort on global memory
	int i = blockIdx.x;
	int j = threadIdx.x + blockIdx.y*blockDim.x;
	int start_idx = j*start_idx_stride1 + i*start_idx_stride2;
	if ((start_idx + sort_stride * (n_data_sort - 1)) >= n_data_total) { return; }
	//selectionSort(src_dst, start_idx, sort_stride, n_data_sort);
	//insertionSort(src_dst, start_idx, sort_stride, n_data_sort);
	//heapSort(src_dst, start_idx, sort_stride, n_data_sort);
	//quickSort(src_dst, start_idx, sort_stride, 0, n_data_sort-1);
	bitonicSortSimple(src_dst, start_idx, sort_stride, n_data_sort);
}


__global__ void sortKernelCustome(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, int algo)
{// sort with custome algo
	int i = blockIdx.x;
	int j = threadIdx.x + blockIdx.y*blockDim.x;
	int start_idx = j*start_idx_stride1 + i*start_idx_stride2;
	if ((start_idx + sort_stride * (n_data_sort - 1)) >= n_data_total) { return; }
	if (algo == 0) { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
	else if (algo == 1) { insertionSort(src_dst, start_idx, sort_stride, n_data_sort); }
	else if (algo == 2) { heapSort(src_dst, start_idx, sort_stride, n_data_sort); }
	else if (algo == 3) { quickSort(src_dst, start_idx, sort_stride, 0, n_data_sort - 1); }
	else if (algo == 4) { bitonicSortVirtualPad(src_dst, start_idx, sort_stride, n_data_sort); }
	else { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
}

__global__ void sortKernelCustome2(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, int algo)
{// use buffer memory
	int i = blockIdx.x;
	int j = threadIdx.x + blockIdx.y*blockDim.x;
	int start_idx = j*start_idx_stride1 + i*start_idx_stride2;
	if ((start_idx + sort_stride * (n_data_sort - 1)) >= n_data_total) { return; }
	float buf[1024];
	for (int p = 0; p < n_data_sort; p++) { buf[p] = src_dst[start_idx + p*sort_stride]; }
	if (algo == 0) { selectionSort(buf, 0, 1, n_data_sort); }
	else if (algo == 1) { insertionSort(buf, start_idx, sort_stride, n_data_sort); }
	else if (algo == 2) { heapSort(buf, 0, 1, n_data_sort); }
	else if (algo == 3) { quickSort(buf, 0, 1, 0, n_data_sort - 1); }
	else if (algo == 4) { bitonicSortVirtualPad(buf, 0, 1, n_data_sort); }
	else { selectionSort(buf, 0, 1, n_data_sort); }
	for (int p = 0; p < n_data_sort; p++) { src_dst[start_idx + p*sort_stride] = buf[p]; }
}

__global__ void sortingNetworkSmallArrayKernel(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, float* virtual_pad)
{// assuming array size less than max permitted block size
	// sort on global memory
	const int start_idx = blockIdx.x*start_idx_stride1 + blockIdx.y*start_idx_stride2;
	const int vp_start_idx = (blockIdx.x*gridDim.y + blockIdx.y)*n_data_sort;
	const int n_round = ceil(log2f(n_data_sort));
	const int n_virtual = powf(2, n_round);
	const int n_missing = n_virtual - n_data_sort;
	const int t_id0 = threadIdx.x;
	const float very_big = 3e38;
	if (n_missing != 0)
	{
		int loc = t_id0;
		if (loc < n_missing)
		{
			virtual_pad[vp_start_idx + loc] = very_big;
		}
	}
	__syncthreads();
	int nbswap0 = 1;
	for (int i = 0; i < n_round; i++)
	{
		int nbswap = nbswap0;
		for (int j = i; j >= 0; j--)
		{
			int n_set = n_virtual / nbswap / 2;
			int k = t_id0 / nbswap;
			int m = t_id0 % nbswap;
			int loc1 = k*nbswap * 2 + m;
			int loc2 = loc1 + nbswap;
			int reverse = (loc1 / nbswap0 / 2) % 2;
			if (loc1 >= n_data_sort)
			{
				bitonicSwap(virtual_pad[vp_start_idx + loc1 - n_data_sort], virtual_pad[vp_start_idx + loc2 - n_data_sort], reverse);
			}
			else if (loc2 >= n_data_sort)
			{
				bitonicSwap(src_dst[start_idx + loc1*sort_stride], virtual_pad[vp_start_idx + loc2 - n_data_sort], reverse);
			}
			else
			{
				bitonicSwap(src_dst[start_idx + loc1*sort_stride], src_dst[start_idx + loc2*sort_stride], reverse);
			}
			__syncthreads();
			nbswap /= 2;
		}
		nbswap0 *= 2;
		//if (i==10)break;
	}
	return;/**/
}

__global__ void sortingNetworkSmallArrayKernel2(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, float* virtual_pad)
{// sort on shared memory
	const int start_idx = blockIdx.x*start_idx_stride1 + blockIdx.y*start_idx_stride2;
	const int vp_start_idx = (blockIdx.x*gridDim.y + blockIdx.y)*n_data_sort;
	const int n_round = ceil(log2f(n_data_sort));
	const int n_virtual = powf(2, n_round);
	const int n_missing = n_virtual - n_data_sort;
	const int t_id0 = threadIdx.x;
	const float very_big = 3e38;
	const int t_idx1 = t_id0;
	const int t_idx2 = t_id0 + blockDim.x;
	__shared__ float buf[4096];
	buf[t_idx1] = src_dst[start_idx + t_idx1*sort_stride];
	if (t_idx2<n_data_sort) { buf[t_idx2] = src_dst[start_idx + t_idx2*sort_stride]; }
	else { buf[t_idx2] = very_big; }
	__syncthreads();
	int nbswap0 = 1;
	for (int i = 0; i < n_round; i++)
	{
		int nbswap = nbswap0;
		for (int j = i; j >= 0; j--)
		{
			int n_set = n_virtual / nbswap / 2;
			int k = t_id0 / nbswap;
			int m = t_id0 % nbswap;
			int loc1 = k*nbswap * 2 + m;
			int loc2 = loc1 + nbswap;
			int reverse = (loc1 / nbswap0 / 2) % 2;
			bitonicSwap(buf[loc1], buf[loc2], reverse);
			nbswap /= 2;
			__syncthreads();
		}
		nbswap0 *= 2;
		//if (i==10)break;
	}
	src_dst[start_idx + t_idx1*sort_stride] = buf[t_idx1];
	if (t_idx2<n_data_sort) { src_dst[start_idx + t_idx2*sort_stride] = buf[t_idx2]; }
	return;/**/
}

__global__ void sortingNetworkLargeArrayKernel(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, float* virtual_pad)
{
	const int start_idx = blockIdx.x*start_idx_stride1 + blockIdx.y*start_idx_stride2;
	const int vp_start_idx = (blockIdx.x*gridDim.y + blockIdx.y)*n_data_sort;
	const int n_round = ceil(log2f(n_data_sort));
	const int n_virtual = powf(2, n_round);
	const int n_missing = n_virtual - n_data_sort;
	const int t_id0 = threadIdx.x;
	const int njob_per_thread = n_data_sort / blockDim.x + (int)((n_data_sort % blockDim.x) != 0);
	const float very_big = 3e38;
	if (n_missing != 0)
	{
		if (blockDim.x >= n_missing)
		{
			int loc = t_id0;
			if (loc < n_missing)
			{
				virtual_pad[vp_start_idx + loc] = very_big;
			}
		}
		else
		{
			int mul = (n_missing / blockDim.x) + 1;
			for (int i = 0; i < mul; i++)
			{
				int loc = blockDim.x*mul + t_id0;
				if (loc < n_missing)
				{
					virtual_pad[vp_start_idx + loc] = very_big;
				}
			}
		}
	}
	/*return;*/
	__syncthreads();
	int nbswap0 = 1;
	for (int i = 0; i < n_round; i++)
	{
		int nbswap = nbswap0;
		for (int j = i; j >= 0; j--)
		{
			int n_set = n_virtual / nbswap / 2;
			for (int p = 0; p < njob_per_thread; p++)
			{
				int t_idn = t_id0 + p*blockDim.x;
				int k = t_idn / nbswap;
				int m = t_idn % nbswap;
				int loc1 = k*nbswap * 2 + m;
				int loc2 = loc1 + nbswap;
				int reverse = (loc1 / nbswap0 / 2) % 2;
				if (loc1 >= n_data_sort)
				{
					bitonicSwap(virtual_pad[vp_start_idx + loc1 - n_data_sort], virtual_pad[vp_start_idx + loc2 - n_data_sort], reverse);
				}
				else if (loc2 >= n_data_sort)
				{
					bitonicSwap(src_dst[start_idx + loc1*sort_stride], virtual_pad[vp_start_idx + loc2 - n_data_sort], reverse);
				}
				else
				{
					bitonicSwap(src_dst[start_idx + loc1*sort_stride], src_dst[start_idx + loc2*sort_stride], reverse);
				}
			}
			__syncthreads();
			nbswap /= 2;
		}
		nbswap0 *= 2;
		//if (i==10)break;
	}
	return;/**/
}


bool sortSimpleCuda(float* const src_dst, char axis, int nx, int ny, int nz, int const cuda_dev_id)
{
	if (!((axis == 'x') || (axis == 'x') || (axis == 'Y') || (axis == 'y') || (axis == 'Z') || (axis == 'z')))
	{
		std::cout << "Invalid axis '" << axis << "' for sorting (only x, y or z)." << std::endl;
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
	float *_d_src_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_src_dst, nx* ny * nz * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src_dst, src_dst, nx* ny * nz * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Launch a kernel on the GPU with one thread for each element.
	if ((axis == 'x') || (axis == 'x'))
	{
		int block_dim = ny;
		int grid_dim = nz;
		int n_data_sort = nx;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = nx;
		int start_idx_stride2 = nx*ny;
		int sort_stride = 1;
		sortKernel2 << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride);
	}
	else if ((axis == 'y') || (axis == 'Y'))
	{
		int block_dim = nx;
		int grid_dim = nz;
		int n_data_sort = ny;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = 1;
		int start_idx_stride2 = nx*ny; 
		int sort_stride = nx;
		//std::cout << "start_idx_stride1 " << start_idx_stride1 << std::endl;
		//std::cout << "start_idx_stride2 " << start_idx_stride2 << std::endl;
		//std::cout << "sort_stride " << sort_stride << std::endl;
		sortKernel2 << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride);
	}
	else if ((axis == 'z') || (axis == 'Z'))
	{
		int block_dim = nx;
		int grid_dim = ny;
		int n_data_sort = nz;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = 1; 
		int start_idx_stride2 = nx;
		int sort_stride = nx*ny;
		sortKernel2 << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride);
	}
	else
	{
		std::cout << "Invalid axis '" << axis << "' for sorting (only x, y or z)." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform sorting, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(src_dst, _d_src_dst, nx* ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
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

bool sortCuda(float* const src_dst, char axis, int nx, int ny, int nz, int algo, int block_size, int const cuda_dev_id)
{
	if (!((axis == 'x') || (axis == 'x') || (axis == 'Y') || (axis == 'y') || (axis == 'Z') || (axis == 'z')))
	{
		std::cout << "Invalid axis '" << axis << "' for sorting (only x, y or z)." << std::endl;
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
	float *_d_src_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_src_dst, nx* ny * nz * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src_dst, src_dst, nx* ny * nz * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Launch a kernel on the GPU with one thread for each element.
	if ((axis == 'x') || (axis == 'x'))
	{
		int block_dim = block_size;
		dim3 grid_dim(nz, (ny / block_size + (int)((ny % block_size)>0)));
		int n_data_sort = nx;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = nx;
		int start_idx_stride2 = nx*ny;
		int sort_stride = 1;
		int mem_virtualpad = n_data_sort*sizeof(float); // for bitonic sort
		sortKernelCustome << <grid_dim, block_dim, mem_virtualpad >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, algo);
	}
	else if ((axis == 'y') || (axis == 'Y'))
	{
		int block_dim = block_size;
		dim3 grid_dim(nz, (nx / block_size + (int)((nx % block_size)>0)));
		int n_data_sort = ny;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = 1;
		int start_idx_stride2 = nx*ny;
		int sort_stride = nx;
		int mem_virtualpad = n_data_sort * sizeof(float); // for bitonic sort
		sortKernelCustome << <grid_dim, block_dim, mem_virtualpad >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, algo);
	}
	else if ((axis == 'z') || (axis == 'Z'))
	{
		int block_dim = block_size;
		dim3 grid_dim(ny, (nx / block_size + (int)((nx % block_size)>0)));
		int n_data_sort = nz;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 =  1;
		int start_idx_stride2 = nx;
		int sort_stride = nx*ny;
		int mem_virtualpad = n_data_sort * sizeof(float); // for bitonic sort
		sortKernelCustome << <grid_dim, block_dim, mem_virtualpad >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, algo);
	}
	else
	{
		std::cout << "Invalid axis '" << axis << "' for sorting (only x, y or z)." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform sorting, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(src_dst, _d_src_dst, nx* ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
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

bool sortingNetworkCuda(float* const src_dst, char axis, int nx, int ny, int nz, int const cuda_dev_id)
{
	if (!((axis == 'x') || (axis == 'x') || (axis == 'Y') || (axis == 'y') || (axis == 'Z') || (axis == 'z')))
	{
		std::cout << "Invalid axis '" << axis << "' for sorting (only x, y or z)." << std::endl;
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
	float *_d_src_dst = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_src_dst, nx* ny * nz * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	float *_d_virtual_pad = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_virtual_pad, nx* ny * nz * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_src_dst, src_dst, nx* ny * nz * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Launch a kernel on the GPU with one thread for each element.
	const int max_block_size = 1024;
	if ((axis == 'x') || (axis == 'x'))
	{
		int n_data_sort = nx;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = nx;
		int start_idx_stride2 = nx*ny;
		int sort_stride = 1;
		int n_round = ceil(log2f(n_data_sort));
		int n_virtual = powf(2, n_round);
		int block_dim = std::min(max_block_size, n_virtual / 2);
		dim3 grid_dim(ny, nz);
		//sortingNetworkLargeArrayKernel << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, _d_virtual_pad);
		sortingNetworkSmallArrayKernel2 << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, _d_virtual_pad);
	}
	else if ((axis == 'y') || (axis == 'Y'))
	{
		int n_data_sort = ny;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = 1;
		int start_idx_stride2 = nx*ny;
		int sort_stride = nx;
		int n_round = ceil(log2f(n_data_sort));
		int n_virtual = powf(2, n_round);
		int block_dim = std::min(max_block_size, n_virtual / 2);
		dim3 grid_dim(nx, nz);
		//sortingNetworkLargeArrayKernel << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, _d_virtual_pad);
		sortingNetworkSmallArrayKernel2 << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, _d_virtual_pad);
	}
	else if ((axis == 'z') || (axis == 'Z'))
	{
		int n_data_sort = nz;
		int n_data_total = nx*ny*nz;
		int start_idx_stride1 = 1;
		int start_idx_stride2 = nx;
		int sort_stride = nx*ny;
		int n_round = ceil(log2f(n_data_sort));
		int n_virtual = powf(2, n_round);
		int block_dim = std::min(max_block_size, n_virtual / 2);
		dim3 grid_dim(nx, ny);
		//sortingNetworkLargeArrayKernel << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, _d_virtual_pad);
		sortingNetworkSmallArrayKernel2 << <grid_dim, block_dim >> >(_d_src_dst, n_data_sort, n_data_total, start_idx_stride1, start_idx_stride2, sort_stride, _d_virtual_pad);
	}
	else
	{
		std::cout << "Invalid axis '" << axis << "' for sorting (only x, y or z)." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to perform sorting, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
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
	cudaStatus = cudaMemcpy(src_dst, _d_src_dst, nx* ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to copy memory from host to device, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	cudaFree(_d_src_dst);
	cudaFree(_d_virtual_pad);
	cudaDeviceReset();
	return true;
}

bool sort2DWithStream(float* const src_dst, int nx, int ny, int const cuda_dev_id)
{// only sort in x-direction
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
	cudaStatus = cudaMalloc((void**)&_d_src_dst, nx* ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}
	float *_d_virtual_pad = nullptr;
	cudaStatus = cudaMalloc((void**)&_d_virtual_pad, nx* ny * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Failed to allocate CUDA memory, " << cudaGetErrorString(cudaStatus) << "." << std::endl;
		cudaDeviceReset();
		return false;
	}

	// declare stream
	const int n_stream = 2;
	cudaStream_t stream_s[n_stream];
	int n_data_sort = nx;
	int n_data_total = nx*ny;
	int start_idx_stride1 = nx;
	int start_idx_stride2 = nx*ny;
	int sort_stride = 1;
	int n_data_stream = nx*ny/n_stream;
	int n_round = ceil(log2f(n_data_sort));
	int n_virtual = powf(2, n_round);
	int block_dim = n_virtual / 2;
	dim3 grid_dim(ny / n_stream, 1);
	for (int i = 0; i < n_stream; i++)
	{
		// create stream
		cudaStreamCreate(&stream_s[i]);

		// copy data async
		cudaMemcpyAsync(&_d_src_dst[n_data_stream*i], &src_dst[n_data_stream*i],
			n_data_stream * sizeof(float), cudaMemcpyHostToDevice, stream_s[i]);

		// perform calculation
		sortingNetworkSmallArrayKernel2 << <grid_dim, block_dim, 0, stream_s[i] >> >(
			&_d_src_dst[n_data_stream*i],
			n_data_sort, n_data_stream, 
			start_idx_stride1, start_idx_stride2, sort_stride, 
			&_d_virtual_pad[n_data_stream*i]);

		// copy results async
		cudaMemcpyAsync(&src_dst[n_data_stream*i], &_d_src_dst[n_data_stream*i],
			n_data_stream * sizeof(float), cudaMemcpyDeviceToHost, stream_s[i]);
	}

	for (int i = 0; i < n_stream; ++i)
	{
		cudaStreamDestroy(stream_s[i]);
	}

	cudaFree(_d_src_dst);
	cudaFree(_d_virtual_pad);
	cudaDeviceReset();
	return true;
}

bool sort(float* const src_dst, char axis, int nx, int ny, int nz, int algo)
{// cpu
	if ((axis == 'x') || (axis == 'x'))
	{
		int sort_stride = 1;
		int n_data_sort = nx;
		for (int i = 0; i < nz; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				int start_idx = i*nx * ny + j*nx;
				if (algo == 0) { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 1) { insertionSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 2) { heapSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 3) { quickSort(src_dst, start_idx, sort_stride, 0, n_data_sort -1); }
				else if (algo == 4) { bitonicSortVirtualPad(src_dst, start_idx, sort_stride, n_data_sort); }
				else { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
			}
		}
		return true;
	}
	else if ((axis == 'y') || (axis == 'Y'))
	{
		int sort_stride = nx;
		int n_data_sort = ny;
		for (int i = 0; i < nz; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				int start_idx = i*nx * ny + j;
				if (algo == 0) { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 1) { insertionSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 2) { heapSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 3) { quickSort(src_dst, start_idx, sort_stride, 0, n_data_sort - 1); }
				else if (algo == 4) { bitonicSortVirtualPad(src_dst, start_idx, sort_stride, n_data_sort); }
				else { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
			}
		}
		return true;
	}
	else if ((axis == 'z') || (axis == 'Z'))
	{
		int sort_stride = nx * ny;
		int n_data_sort = nz;
		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				int start_idx = i * nx + j;
				//std::cout << i << ", " << j << ", " << start_idx << std::endl;
				if (algo == 0) { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 1) { insertionSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 2) { heapSort(src_dst, start_idx, sort_stride, n_data_sort); }
				else if (algo == 3) { quickSort(src_dst, start_idx, sort_stride, 0, n_data_sort - 1); }
				else if (algo == 4) { bitonicSortVirtualPad(src_dst, start_idx, sort_stride, n_data_sort); }
				else { selectionSort(src_dst, start_idx, sort_stride, n_data_sort); }
			}
		}
		return true;
		//std::cout << "done" << std::endl;
	}
	else
	{
		std::cout << "Invalid axis '" << axis << "' for sorting (only x, y or z)." << std::endl;
		return false;
	}
}

