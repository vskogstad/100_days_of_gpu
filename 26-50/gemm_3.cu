#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<int T>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // naive to high performance implementation, Following along with https://siboehm.com/articles/22/CUDA-MMM
    // This is my version of kernel 3. Prefer my indexing. Using smem and tiled read/write to memory. Gradually building up temp until we can write to output.
    int BLOCKSIZE = T;
    const int threadCol = (threadIdx.x % BLOCKSIZE);
    const int threadRow = (threadIdx.x / BLOCKSIZE);
    const int col = blockIdx.x * BLOCKSIZE + threadCol;
    const int row = blockIdx.y * BLOCKSIZE + threadRow;
    const int idx = row * N + col;
    // Setting up tiles and loading to smem:
    int numTiles = (T + K - 1) / T;
    __shared__ half A_s[T][T];
    __shared__ half B_s[T][T];

    float acc = 0.0f;
    for (int t = 0; t < numTiles; t++) {
        // load blocks into smem
        int Acol = (t * T + threadCol);
        int Brow = (t * T + threadRow);
        A_s[threadRow][threadCol] = (Acol < K && row < M) ? A[row * K + Acol] : __half(0.0);
        //printf("bidx =%i: A_s idx = %i | A idx = %i\n", bidx, threadRow * T + threadCol, row * K + (bidx * BLOCKSIZE + threadCol));
        B_s[threadRow][threadCol] = (col < N && Brow < K) ? B[Brow * N + col] : __half(0.0);
        __syncthreads();
        //for (int j = 0; j < K; j++) printf("row =%i: As1 = %f\n", j, A_s[0 + j]);
        // calculate dot_product for the two blocks
        for (int j = 0; j < BLOCKSIZE; j++) {
            acc +=  __half2float((A_s[threadRow][j] * B_s[j][threadCol]));
        }
        __syncthreads();

    }
    //write accumulated result for thread
    if (row < M && col < N){
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