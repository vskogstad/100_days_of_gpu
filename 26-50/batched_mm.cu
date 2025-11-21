#include <cuda_runtime.h>
#define CEIL_DIV(A, B) ((A + B - 1) / B)

template<int T, int Mc, int Mr>
__global__ void batchedMatmul(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int tid = threadIdx.x;
    int tilesPerRow = (T/Mc);
    int tc = tid % tilesPerRow;
    int tr = tid / tilesPerRow;
    int col0 = blockIdx.x * T + tc*Mc;
    int row0 = blockIdx.y * T + tr*Mr;
    int b = blockIdx.z;

    //int C_idx = b * (M*N) + row0 * N + col0;

    // setting up shared memory:
    __shared__ float A_s[T][T+1];
    __shared__ float B_s[T][T+1];

    // defining accumulators for mini-tile
    float acc[Mr][Mc];
    for (int i = 0; i < Mr; i++) {
        for (int j = 0; j < Mc; j++) acc[i][j] = 0.0f;
    }

    int numTiles = CEIL_DIV(K, T);
    // read tiles of A and B into smem
    for (int t = 0; t < numTiles; t++) {
        int Acol = t*T + tc*Mc;
        int Brow = t*T + tr*Mr;
        // load each mini-tile into A_s and B_s
        for (int r = 0; r < Mr; r++) {
            for(int c = 0; c < Mc; c++) {
                int A_idx = b * (M*K) + (row0 + r) * K + Acol + c;
                int B_idx = b * (N*K) + (Brow + r) * N + col0 + c;
                int tileRow = tr * Mr + r;
                int tileCol = tc * Mc + c;
                A_s[tileRow][tileCol] = (Acol + c < K && row0 + r < M) ? A[A_idx] : 0.0f;
                B_s[tileRow][tileCol] = (col0 + c < N && Brow + r < K) ? B[B_idx] : 0.0f;

            }

        }
        __syncthreads();
        // add current tiles contibution to dot product 
        for (int i = 0; i < T; i++) {
            for (int r = 0; r < Mr; r++) {
                float A = A_s[tr*Mr+r][i];
                for(int c = 0; c < Mc; c++) {
                    acc[r][c] += A * B_s[i][tc*Mc+c];
                }

            }
        }
        __syncthreads();
    }

    // write result to C
    for (int r = 0; r < Mr; r++) {
        for(int c = 0; c < Mc; c++) {
            if (row0 + r < M && col0 + c < N) {
                int C_idx = b * (M*N) + (row0 + r) * N + col0 + c;
                C[C_idx] = acc[r][c];
            }   
        }
    }


}


// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int T = 32;
    int Mc = 8, Mr = 4;
    int numThreads = (T/Mc)*(T/Mr);
    dim3 numBlocks(CEIL_DIV(N, T), CEIL_DIV(M,T), BATCH);
    batchedMatmul<32, 8, 4><<<numBlocks, numThreads>>>(A, B, C, BATCH, M, N, K);

} 