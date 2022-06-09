#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

__global__ void transposeKernel(float* dst, float* src, int nx, int ny);

bool transposeCuda(float* const src_dst, int const nx, int const ny, int const cuda_dev_id = 0);

// cpu
void transpose(float* const src_dst, int const nx, int const ny);

