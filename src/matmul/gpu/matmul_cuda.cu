#include <cuda_runtime.h>
#include <stdio.h> // 确保包含这个头文件
#include <math.h>
#include "error_check.h"

const int block_M = 16;
const int block_N = 16;

__global__ void matmul_kernel_fp32(
    float *device_A,
    float *device_B,
    float *device_C,
    int M, int K, int N)
{ // A: M*K B: K*N C: M*N
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp = 0;
    if (y < N && x < M)
    {
        for (int i = 0; i < K; i++)
        {
            tmp += device_A[K * x + i] * device_B[N * i + y];
        }
        device_C[x * N + y] = tmp;
    }
}

extern "C" void matmul_cuda_fp32(void *const host_A,
                                 void *const host_B,
                                 void *const host_C,
                                 const int M, const int K, const int N)
{
    int num_block_x = (M) / (block_M) + 1;
    int num_block_y = (N) / (block_N) + 1;
    dim3 grid_dim(num_block_x, num_block_y);
    dim3 block_dim(block_M, block_N);
    matmul_kernel_fp32<<<grid_dim, block_dim>>>((float *)host_A,
                                                (float *)host_B,
                                                (float *)host_C,
                                                M, K, N);
}