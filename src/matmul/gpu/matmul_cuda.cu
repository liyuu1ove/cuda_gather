#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "error_check.h"


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


static const int block_size = 32;
template <int BLOCKSIZE>
__global__ void matmul_kernel_fp32(
    const float *device_A,
    const float *device_B,
    float *device_C,
    int M, int K, int N)// A: M*K B: K*N C: M*N
{            
    int col_C = blockIdx.x*blockDim.x; //index of C element to compute
    int row_C = blockIdx.y*blockDim.y;
    
    int start_A = K*row_C; //index of startpoint in A in a block
    int start_B = col_C; //index of startpoint in B in a block
    int index_A = start_A+threadIdx.x+threadIdx.y*K;
    int index_B = start_B+threadIdx.x+threadIdx.y*N;
    
    __shared__ float shared_A[BLOCKSIZE*BLOCKSIZE];
    __shared__ float shared_B[BLOCKSIZE*BLOCKSIZE];
    float tmp=0;
    
    for(int idx_K =0 ;idx_K<K;idx_K+=BLOCKSIZE){
        int tid=threadIdx.x+threadIdx.y*BLOCKSIZE;//tid of the block
        shared_A[tid]=device_A[index_A];
        shared_B[tid]=device_B[index_B];
        __syncthreads();
        for(int i =0 ;i<BLOCKSIZE;i++){
            tmp+=shared_A[threadIdx.y*BLOCKSIZE+i]*shared_B[threadIdx.x+i*BLOCKSIZE];
        }
        __syncthreads();
        index_A+=BLOCKSIZE;
        index_B+=BLOCKSIZE*N;
    }
    device_C[col_C+threadIdx.x+(row_C+threadIdx.y)*N]=tmp;
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
    int num_block_y = CEIL_DIV(M, block_size);
    int num_block_x = CEIL_DIV(N, block_size);
    dim3 grid_dim(num_block_x, num_block_y);
    dim3 block_dim(block_size, block_size);
    matmul_kernel_fp32<block_size><<<grid_dim, block_dim>>>((float *)host_A,
                                                            (float *)host_B,
                                                            (float *)host_C,
                                                            M, K, N);
}