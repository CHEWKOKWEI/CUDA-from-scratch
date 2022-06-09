#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

__global__ void convolve2dKernel(float *dst, float *src, float *kernel, int src_nx, int src_ny, int kern_nx, int kern_ny);

__global__ void convolve2dKernel2(float *dst, float *src, float *kernel, int src_nx, int src_ny, int kern_nx, int kern_ny);

__global__ void convolveFull2dKernel(float *dst, float *src, float *kernel, int src_nx, int src_ny, int kern_nx, int kern_ny);


bool convolve2dCuda(float* const dst, float *const src, float *const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny, int const cuda_dev_id = 0);

bool convolve2dCuda2(float* const dst, float *const src, float *const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny, int const cuda_dev_id = 0);

bool convolveFull2dCuda(float* const dst, float *const src, float *const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny, int block_size = 64, int const cuda_dev_id = 0);


// cpu
void convolve2d(float* const dst, float* const src, float* const kernel, int const src_nx, int const src_ny, int const kern_nx, int const kern_ny);
