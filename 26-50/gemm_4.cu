#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<int tc, int T>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // naive to high performance implementation, Following along with https://siboehm.com/articles/22/CUDA-MMM
    // This is my version of kernel 4. Prefer my indexing. Using smem and tiled read/write to memory. Gradually building up temp until we can write to output.
    int BLOCKSIZE = T;
    const int threadCol = (threadIdx.x % (T/tc));
    const int threadRow = (threadIdx.x / (T/tc));
    const int col0 = blockIdx.x * T + threadCol*tc;
    const int row = blockIdx.y * BLOCKSIZE + threadRow;
    const int idx0 = row * N + col0;
    // Setting up tiles and loading to smem:
    int numTiles = (T + K - 1) / T;
    __shared__ float A_s[T][T];
    __shared__ float B_s[T][T];
    
    // setting up accumulators in registers
    float acc[tc];
    for (int n = 0; n < tc; n++) acc[n] = 0.0f;
    //float acc1 = 0.0f;


    for (int t = 0; t < numTiles; t++) {
        // load blocks into smem
        int Acol = (t * T + threadCol*tc);
        int Brow = (t * T + threadRow);
        // loading in all cols in mini-tile
        for (int n = 0; n < tc; n++) {
            A_s[threadCol*tc + n][threadRow] = (Acol + n < K && row < M) ? __half2float(A[row * K + Acol + n]) : 0.0f;
            B_s[threadRow][threadCol*tc + n] = (col0 + n < N && Brow < K) ? __half2float(B[Brow * N + col0+n]) : 0.0f;
        }
        //A_s[threadRow][threadCol*tc] = (Acol < K && row < M) ? __half2float(A[row * K + Acol]) : 0.0f;
        //B_s[threadRow][threadCol*tc] = (col0 < N && Brow < K) ? __half2float(B[Brow * N + col0]) : 0.0f;
        // second elements
        //int n = 1; // 
        
        __syncthreads();

        for (int j = 0; j < BLOCKSIZE; j++) {
            // load A for mini-tile into register:
            float A = A_s[j][threadRow];
            // update all accumulators in register
            #pragma unroll
            for (int n = 0; n < tc; n++) {
                acc[n] +=  (A * B_s[j][threadCol*tc+n]);
            }
            //acc0 +=  (A * B_s[j][threadCol*tc]);
            
        }
        __syncthreads();

        

    }
    //write accumulated result for thread
    if (row < M && col0 < N){
        for (int n = 0; n < tc; n++) {
            if (col0 + n < N) C[idx0+n] = __float2half(alpha * acc[n] + beta * __half2float(C[idx0+n]));
        }
        
    }
}    




// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int tc = 4;
    int T = 32;
    dim3 threadsPerBlock(T*T/tc);
    dim3 numBlocks(((T + N - 1) / T), (T + M - 1) / T);  // x, y
    gemm<4, 32><<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}