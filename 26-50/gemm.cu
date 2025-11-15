#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // naive to high performance implementation, Following along with https://siboehm.com/articles/22/CUDA-MMM eventually
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int xi = blockIdx.x * blockDim.x + tx;
    int yi = blockIdx.y * blockDim.y + ty;
    int idx = yi * N + xi;
    if (yi < M && xi < N){
        float acc = 0.0f;
        for (int j = 0; j < K; j++) {
            acc +=  __half2float((A[yi*K + j] * B[N*j + xi]));
        }
        C[idx] = alpha *acc + beta * __half2float(C[idx]);
    }
    

}


// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock = (32,32);
    dim3 numBlocks = ((threadsPerBlock.y + M - 1) / threadsPerBlock.y,(threadsPerBlock.x + N - 1) / threadsPerBlock.x);
    gemm<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
