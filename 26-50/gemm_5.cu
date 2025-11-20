#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<int tc, int tr, int T>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // naive to high performance implementation, Following along with https://siboehm.com/articles/22/CUDA-MMM
    // This is my version of kernel 5 wit 2d-tiling. Using smem and tiled read/write to memory. 
    int BLOCKSIZE = T;
    int tilesPerRow = T / tc;
    const int threadCol = threadIdx.x % tilesPerRow;
    const int threadRow = threadIdx.x / tilesPerRow;
    const int col0 = blockIdx.x * T + threadCol * tc;
    const int row0 = blockIdx.y * T + threadRow * tr;
    const int idx0 = row0 * N + col0;
    // Setting up tiles and loading to smem:
    int numTiles = (T + K - 1) / T;
    __shared__ float A_s[T][T];
    __shared__ float B_s[T][T];
    
    // setting up accumulators in registers
    float acc[tr][tc];
    for (int c = 0; c < tc; c++) {
        for (int r = 0; r < tr; r++) acc[r][c] = 0.0f;
    }

    for (int t = 0; t < numTiles; t++) {
        // load blocks into smem
        int Acol = (t * T + threadCol*tc);
        int Brow = (t * T + threadRow*tr);

        // loading in for each row in mini-tile:
        for (int r = 0; r < tr; r++) {
            // for each col in the mini-tile:
            #pragma unroll
            for (int c = 0; c < tc; c++) {
                // Transposed A_s for faster reads
                A_s[threadCol*tc + c][threadRow*tr + r] = (Acol + c < K && row0 + r < M) ? __half2float(A[(row0 + r) * K + Acol + c]) : 0.0f;
                B_s[threadRow*tr + r][threadCol*tc + c] = (col0 + c < N && Brow + r < K) ? __half2float(B[(Brow + r) * N + col0 + c]) : 0.0f;
            }
        }
        
        __syncthreads();

        for (int j = 0; j < BLOCKSIZE; j++) {
            // loading B_S for all values of c into registers here first. No real speedup, compiler likely already doing this
            float B[tc];
            for (int c = 0; c < tc; c++) {
                    B[c] = B_s[j][threadCol*tc+c];
                }

            for (int r = 0; r < tr; r++) {
                // load A for mini-tile into register:
                float A = A_s[j][threadRow*tr + r];
                // update all accumulators in register
                #pragma unroll
                for (int c = 0; c < tc; c++) {
                    acc[r][c] += (A * B[c]);
                }
            }
            
            //acc0 +=  (A * B_s[j][threadCol*tc]);
            
        }
        __syncthreads();

        

    }
    //write accumulated result for thread
    if (row0 < M && col0 < N){
        for (int r = 0; r < tr; r++) {
            if (row0 + r < M) {
                for (int c = 0; c < tc; c++) {
                    if (col0 + c < N) C[idx0+c + N*r] = __float2half(alpha * acc[r][c] + beta * __half2float(C[idx0+c + N*r]));
                }
            }    
        }
    }
}    




// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int tc = 4;
    int tr = 8;
    int T = 32;
    dim3 threadsPerBlock((T/tc)*(T/tr));
    dim3 numBlocks(((T + N - 1) / T), (T + M - 1) / T);  // x, y
    gemm<4, 8, 32><<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}