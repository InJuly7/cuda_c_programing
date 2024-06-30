#pragma once
#include <stdio.h>
#include <cuda_runtime.h>


// 二维网格 二维线程块
// 每个thread 处理一个 元素
__global__ void MatrixSum_grid2d_block2d(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix; 
    if (ix < nx && iy < ny) 
        MatC[idx] = MatA[idx] + MatB[idx];

}

__global__ void MatrixSum_grid1d_block1d(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
    if(ix < nx)
    {
        for(int iy = 0; iy < ny; iy++)
        {
            int idx = iy*nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

__global__ void MatricSum_grid2d_block1d(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy*nx + ix; 
    if (ix < nx && iy < ny) 
        MatC[idx] = MatA[idx] + MatB[idx];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{ 
    int i = threadIdx.x;
    C[i] = A[i] + B[i]; 
}


