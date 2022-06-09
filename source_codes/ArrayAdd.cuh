#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

__global__ void addKernel1D(float *dst, float *src1, float *src2, int n_data);

bool arrayAddCuda(float* const dst, float *const src1, float *const src2, int const n_data, int const cuda_dev_id);

void arrayAdd(float* const dst, float* const src1, float* const src2, int const n_data); 
