#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

__device__ __host__ void selectionSort(float* arr, int start_idx, int stride, int n_data);

__device__ __host__ void insertionSort(float* arr, int start_idx, int stride, int n_data);

__device__ __host__ void buildMaxHeap(float* arr, int start_idx, int stride, int n_data);

__device__ __host__ void heapSort(float* arr, int start_idx, int stride, int n_data);

__device__ __host__ void partition(float* arr, int start_idx, int stride, int low, int high, int& partition_index);

__device__ __host__ void quickSort(float* arr, int start_idx, int stride, int low, int high);

__device__ __host__ void bitonicSwap(float &src1, float &src2, int reverse);

__device__ __host__ void bitonicSortSimple(float* arr, int start_idx, int stride, int n_data);

__device__ __host__ void bitonicSortVirtualPad(float* arr, int start_idx, int stride, int n_data);



__global__ void sortKernel(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride);

__global__ void sortKernelCustome(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, int algo = 0);

__global__ void sortingNetworkSmallArrayKernel(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, float* virtual_pad = 0);

__global__ void sortingNetworkSmallArrayKernel2(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, float* virtual_pad = 0);

__global__ void sortingNetworkLargeArrayKernel(float *src_dst, int n_data_sort, int n_data_total, int start_idx_stride1, int start_idx_stride2, int sort_stride, float* virtual_pad=0);


bool sortSimpleCuda(float* const src_dst, char axis, int nx, int ny, int nz, int const cuda_dev_id=0);

bool sortCuda(float* const src_dst, char axis, int nx, int ny, int nz, int algo = 0, int block_size = 1024, int const cuda_dev_id = 0);

bool sortingNetworkCuda(float* const src_dst, char axis, int nx, int ny, int nz, int const cuda_dev_id = 0);

bool sort2DWithStream(float* const src_dst, int nx, int ny, int const cuda_dev_id = 0);


// cpu
bool sort(float* const src_dst, char axis, int nx, int ny, int nz, int algo = 0);


