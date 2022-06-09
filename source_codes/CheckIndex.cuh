#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

__global__ void checkIndexKernel1D_1D(int *thread_id, int *block_id, int *global_id, int n_data);

__global__ void checkIndexKernel2D_2D(int* thread_id_x, int* thread_id_y, int* block_id_x, int* block_id_y, int*  global_id, int n_data);


bool checkIndex1Dgrid_1Dblock();

bool checkIndex2Dgrid_2Dblock();



