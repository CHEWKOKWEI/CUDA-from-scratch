#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

__global__ void flipHorizontalKernel(float* src_dst, int nx, int ny);

__global__ void flipVerticalKernel(float* src_dst, int nx, int ny);


bool flipHorizontalCuda(float* const src_dst, int const nx, int const ny, int const cuda_dev_id = 0);

bool flipVerticalCuda(float* const src_dst, int const nx, int const ny, int const cuda_dev_id = 0);


// cpu
void flipHorizontal(float* const src_dst, int const nx, int const ny);

void flipVertical(float* const src_dst, int const nx, int const ny);

