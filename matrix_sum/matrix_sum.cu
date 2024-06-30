#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include "../checkerror/error.cuh"
#include "../verify_kernel/verify_kernel.h"
#include "../kernel/kernel.cuh"

#define RUN_KERNEL 3
int main(int argc, char **argv)
{ 
    printf("%s Starting...\n", argv[0]); 
    // set up device 
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev)); 
    // set up date size of matrix 
    int nx = 1<<14; 
    int ny = 1<<14; 
    int nxy = nx*ny; 
    int nBytes = nxy * sizeof(float); 
    printf("Matrix size: nx %d ny %d\n",nx, ny); 
    
    float *h_A, *h_B, *hostRef, *gpuRef; 
    h_A = (float *)malloc(nBytes); 
    h_B = (float *)malloc(nBytes); 
    hostRef = (float *)malloc(nBytes); 
    gpuRef = (float *)malloc(nBytes); 
    
    auto start_1 = std::chrono::high_resolution_clock::now();
    initialData (h_A, nxy); 
    initialData (h_B, nxy); 
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
    std::cout << "initialData time: " << elapsed_1.count() << " milliseconds" << std::endl;
   
    memset(hostRef, 0, nBytes); 
    memset(gpuRef, 0, nBytes); 
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    sumMatrixOnHost (h_A, h_B, hostRef, nx,ny); 
    auto end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_2 = end_2 - start_2;
    std::cout << "sumMatrixOnHost time: " << elapsed_2.count() << " milliseconds" << std::endl;


    float *d_MatA, *d_MatB, *d_MatC; 
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes); 
    cudaMalloc((void **)&d_MatC, nBytes); 
    
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); 

    int blockdim_x[] = {1, 16, 32, 64, 128, 256, 512, 1024};
    int blockdim_y[] = {1, 16, 32, 64, 128, 256, 512, 1024};
    MatrixSum_grid2d_block2d <<< 16, 16 >>>(d_MatA, d_MatB, d_MatC, nx, ny);

#if RUN_KERNEL == 1
// grid2D block2D
    for(int i = 0;i < sizeof(blockdim_x) / sizeof(blockdim_x[0]); i++)
    {
        for(int j = 0;j < sizeof(blockdim_y)/sizeof(blockdim_y[0]); j++)
        {
            dim3 block(blockdim_x[i], blockdim_y[j]); 
            dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

            if(!((block.x*block.y*block.z <= 1024) && 
                (block.x<=1024) && 
                (block.y<=1024) && 
                (block.z<=64) && 
                (grid.y<=65535) &&
                (grid.z<=65535))) continue;
            
            auto start_3 = std::chrono::high_resolution_clock::now();
            MatrixSum_grid2d_block2d <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny); 
            cudaDeviceSynchronize();
            auto end_3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_3 = end_3 - start_3;
            printf("MatrixSum_grid2d_block2d <<<(%d,%d), (%d,%d)>>> elapsed %f milliseconds  ", 
                                                            grid.x, grid.y, block.x, block.y, elapsed_3.count()); 
            cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost); 
            checkResult(hostRef, gpuRef, nxy); 
        }
    }
#elif RUN_KERNEL == 2
// grid1D block1D 
    for(int i = 0;i < sizeof(blockdim_x) / sizeof(blockdim_x[0]); i++)
    {
            dim3 block(blockdim_x[i],1); 
            dim3 grid((nx+block.x-1)/block.x,1);

            if(!((block.x*block.y*block.z <= 1024) && 
                (block.x<=1024) && 
                (block.y<=1024) && 
                (block.z<=64) && 
                (grid.y<=65535) &&
                (grid.z<=65535))) continue;
            
            auto start_3 = std::chrono::high_resolution_clock::now();
            MatrixSum_grid1d_block1d <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny); 
            cudaDeviceSynchronize();
            auto end_3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_3 = end_3 - start_3;
            printf("MatrixSum_grid1d_block1d <<<(%d,%d), (%d,%d)>>> elapsed %f milliseconds  ", 
                                                            grid.x, grid.y, block.x, block.y, elapsed_3.count()); 
            cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost); 
            checkResult(hostRef, gpuRef, nxy); 
        }
#elif RUN_KERNEL == 3
// grid2D block1D
    for(int i = 0;i < sizeof(blockdim_x) / sizeof(blockdim_x[0]); i++)
    {
        dim3 block(blockdim_x[i], 1); 
        dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
        if(!((block.x*block.y*block.z <= 1024) && 
            (block.x<=1024) && 
            (block.y<=1024) && 
            (block.z<=64) && 
            (grid.y<=65535) &&
            (grid.z<=65535))) continue;
        
        auto start_3 = std::chrono::high_resolution_clock::now();
        MatrixSum_grid2d_block2d <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny); 
        cudaDeviceSynchronize();
        auto end_3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_3 = end_3 - start_3;
        printf("MatrixSum_grid2d_block2d <<<(%d,%d), (%d,%d)>>> elapsed %f milliseconds  ", 
                                                        grid.x, grid.y, block.x, block.y, elapsed_3.count()); 
        cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost); 
        checkResult(hostRef, gpuRef, nxy); 
    }
#endif
    // cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost); 
    // checkResult(hostRef, gpuRef, nxy); 
    
    cudaFree(d_MatA); 
    cudaFree(d_MatB); 
    cudaFree(d_MatC); 
    
    free(h_A); 
    free(h_B); 
    free(hostRef); 
    free(gpuRef); 
    
    cudaDeviceReset(); 
    return (0); 
}