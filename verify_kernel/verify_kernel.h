#pragma once
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <random>
 
void checkResult(float *hostRef, float *gpuRef, const int N)
{ 
    double epsilon = 1.0E-8;
    int match_flag = 1; 
    for(int i = 0; i < N; i++)
    { 
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) 
        { 
            match_flag = 0; 
            printf("Arrays do not match!\n"); 
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i); break; 
        } 
    }
    
    if (match_flag) printf("Arrays match.\n"); 
    return; 
}

void initialData(float *ip,int size)
{ 
    // 使用当前时间作为种子初始化随机数生成器 
    time_t t; 
    srand((unsigned) time(&t)); 
    for (int i=0; i<size; i++)
    { 
        // ip[i] = (float)( rand() & 0xFF )/10.0f; 
        ip[i] = (i+1)%10;
    } 
}
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{ 
    for (int idx=0; idx<N; idx++) 
    {
        C[idx] = A[idx] + B[idx];
    }
}

void sumMatrixOnHost (float *A, float *B, float *C, const int nx, const int ny) 
{ 
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int iy=0; iy<ny; iy++) 
    { 
        for (int ix=0; ix<nx; ix++)
        { 
            ic[ix] = ia[ix] + ib[ix]; 
        } 
        ia += nx; 
        ib += nx; 
        ic += nx; 
    }
}
