#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "error_check.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
// a thread compute TM*TN elements of C
// and a block compute BM*BN elements of C
// SMEM loads BM*BK + BK*BN elements then compute
// register hold TM*1 and 1*TN elements to compute a part of every element in TM*TN
__global__ void matmul_kernel_fp32(
    const float *device_A,
    const float *device_B,
    float *device_C,
    int M, int K, int N) // A: M*K B: K*N C: M*N
{
    int col_C = (blockIdx.x * blockDim.x + threadIdx.x) * TN; // start index of C element to compute
    int row_C = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    int start_idx_A = blockIdx.y * blockDim.y * TM*K;
    int start_idx_B = blockIdx.x * TN * blockDim.x;
    __shared__ float shared_A[BM * BK];
    __shared__ float shared_B[BK * BN];
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // tid in the block
    int col_shared_A = tid % BK;
    int row_shared_A = tid / BK;
    int col_shared_B = tid % BN;
    int row_shared_B = tid / BN;
    float tmp[TN * TM] = {0};

    for (int idx_K = 0; idx_K < K; idx_K += BK)
    {
        for (int i = 0; i < BM; i += (blockDim.y * blockDim.x / BK)) // BM=TM*blockDim.y load shared_A
        {
            shared_A[col_shared_A + (row_shared_A + i) * BK] = device_A[start_idx_A + col_shared_A + (row_shared_A + i) * K];
        }
        for (int i = 0; i < BK; i += (blockDim.y * blockDim.x / BN)) // BM=TM*blockDim.y load shared_A
        {
            shared_B[col_shared_B + (row_shared_B + i) * BN] = device_B[start_idx_B + col_shared_B + (row_shared_B + i) * N];
        }

        __syncthreads();
        for (int i = 0; i < BK; i++)
        {
            for (int k = 0; k < TM; k++)
            {
                for (int j = 0; j < TN; j++)
                {

                    tmp[j + k * TN] += shared_A[(threadIdx.y * TM + k) * BK + i] * shared_B[threadIdx.x * TN + j + i * BN];
                }
            }
        }
        __syncthreads();
        start_idx_A += BK;
        start_idx_B += BK * N;
    }
    for (int k = 0; k < TM; k++)
    {
        for (int j = 0; j < TN; j++)
        {

            device_C[col_C + j + (row_C + k) * N] = tmp[j + k * TN];
        }
    }
}
//__global__ void matmul_tensorcore_
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
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TN = 8;
    const int TM = 8;

    int num_block_y = CEIL_DIV(M, BM);
    int num_block_x = CEIL_DIV(N, BN);
    dim3 grid_dim(num_block_x, num_block_y);
    dim3 block_dim(CEIL_DIV(BN, TN), CEIL_DIV(BM, TM));
    matmul_kernel_fp32<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>((float *)host_A,
                                                                    (float *)host_B,
                                                                    (float *)host_C,
                                                                    M, K, N);
}