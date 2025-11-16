#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<int T>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // naive to high performance implementation, Following along with https://siboehm.com/articles/22/CUDA-MMM
    // This is my version of kernel 3. Prefer my indexing. Using smem and tiled read/write to memory. Gradually building up temp until we can write to output.
    int BLOCKSIZE = T;
    const int col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    const int row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int idx = row * N + col;
    // Setting up tiles and loading to smem:
    __shared__ half* A_s[T][T];
    __shared__ half* B_s[T][T];

    for (int t = t0; t < t_max; t+= T*T) {
        //A_s[col][row] = A[row*]

    //

        if (row < M && col < N){
            float acc = 0.0f;
            for (int j = 0; j < K; j++) {
                acc +=  __half2float((A[row*K + j] * B[N*j + col]));
            }
        }
    //printf("%i: %f\n", idx, acc);
    C[idx] = __float2half(alpha * acc + beta * __half2float(C[idx]));
    
}
        

}


// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int T = 32;
    dim3 threadsPerBlock(T*T);
    dim3 numBlocks((T + N - 1) / T, (T + M - 1) / T);  // x, y
    gemm<32><<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
