#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "error_check.h"

const int block_size = 16;
template <int BLOCKSIZE>
__global__ void matmul_kernel_fp32(
    float *device_A,
    float *device_B,
    float *device_C,
    int M, int K, int N)
{                                                      // A: M*K B: K*N C: M*N
    int index = threadIdx.x + threadIdx.y * BLOCKSIZE; // index in a block
    int x = blockIdx.x * BLOCKSIZE + index / (BLOCKSIZE);
    int y = blockIdx.y * BLOCKSIZE + index % (BLOCKSIZE);

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
extern "C" void matmul_cudnn_fp32(void *const host_A,
                                  void *const host_B,
                                  void *const host_C,
                                  const int M, const int K, const int N)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, (float *)host_B, N, (float *)host_A, K, &beta, (float *)host_C, N);
}

extern "C" void matmul_cuda_fp32(void *const host_A,
                                 void *const host_B,
                                 void *const host_C,
                                 const int M, const int K, const int N)
{
    int num_block_x = (M) / (block_size) + 1;
    int num_block_y = (N) / (block_size) + 1;
    dim3 grid_dim(num_block_x, num_block_y);
    dim3 block_dim(block_size, block_size);
    matmul_kernel_fp32<block_size><<<grid_dim, block_dim>>>((float *)host_A,
                                                            (float *)host_B,
                                                            (float *)host_C,
                                                            M, K, N);
}