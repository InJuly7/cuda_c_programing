#include <cuda_runtime.h>
#include <stdio.h>

#include "../checkerror/error.cuh"
#include "../verify_kernel/verify_kernel.h"
#include "../kernel/kernel.cuh"



int main(int argc, char **argv) 
{ 
    printf("%s Starting...\n", argv[0]); 
    int dev = 0; 
    cudaSetDevice(dev); 
    int nElem = 64; 
    printf("Vector size %d\n", nElem); 

    size_t nBytes = nElem * sizeof(float); 
    float *h_A, *h_B, *hostRef, *gpuRef; 
    h_A = (float *)malloc(nBytes); 
    h_B = (float *)malloc(nBytes); 
    hostRef = (float *)malloc(nBytes); 
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem); 
    initialData(h_B, nElem); 
    memset(hostRef, 0, nBytes); 
    memset(gpuRef, 0, nBytes); 
    
    float *d_A, *d_B, *d_C; 
    cudaMalloc((float**)&d_A, nBytes); 
    cudaMalloc((float**)&d_B, nBytes); 
    cudaMalloc((float**)&d_C, nBytes); 
   
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice); 
    
    dim3 block (nElem); 
    dim3 grid (nElem/block.x); 
    sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C); 
    printf("Execution configuration <<<%d, %d>>>\n",grid.x,block.x); 
    
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost); 
     
    sumArraysOnHost(h_A, h_B, hostRef, nElem); 
    // check device results 
    checkResult(hostRef, gpuRef, nElem); 
    // free device global memory 
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C); 
    
    free(h_A); 
    free(h_B); 
    free(hostRef); 
    free(gpuRef); 
    return(0); 
}