#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

__device__ __host__ float trilinearInterpolate(float *src, int src_nx, int src_ny, int src_nz, float x, float y, float z);

__global__ void slicingKernel(float *dst, int dst_nx, int dst_ny,
	float *src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float *angle_s);

__global__ void slicingWithTextureKernel(float *dst, int dst_nx, int dst_ny,
	cudaTextureObject_t tex_src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float *angle_s);

bool slicing(float *dst, int n_angle, int dst_nx, int dst_ny,
	float* const src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float* angle_s, int const cuda_dev_id = 0);

bool slicingWithTexture(float *dst, int n_angle, int dst_nx, int dst_ny,
	float* const src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float* angle_s, int const cuda_dev_id = 0);

bool slicingWithTextureAndStream(float *dst, int n_angle, int dst_nx, int dst_ny,
	float* const src, int src_nx, int src_ny, int src_nz,
	float x0, float y0, float z0, float* angle_s, int const cuda_dev_id = 0);

