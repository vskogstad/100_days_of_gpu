#include <cuda_runtime.h>
//#include <math.h>
#include <stdio.h>

template<int T>
__global__ void quant_mm(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    //int T = 32;
    int tid = threadIdx.x;
    int tc = tid % T;
    int tr = tid / T;
    int col = blockIdx.x * T + tc;
    int row = blockIdx.y * T + tr;

    // load tile to smem:
    __shared__ int A_s[T][T];
    __shared__ int B_s[T][T];

    int numTiles = (K + T - 1) / T;

    int acc = 0;
    for (int t = 0; t < numTiles; t++) {
        int Acol = tc + t*T;
        int Brow = tr + t*T;
        // convert and subract one time, when loading into smem
        A_s[tr][tc] = (Acol < K && row < M) ? static_cast<int>(A[row * K + Acol]) - zero_point_A : 0;
        B_s[tr][tc] = (col < N && Brow < K) ? static_cast<int>(B[Brow * N + col]) - zero_point_B : 0;
        __syncthreads();

        // Find dot product 
        if (col < N && row < M) {
            for (int i = 0; i < T; i++) {
                acc += A_s[tr][i] * B_s[i][tc];
            }
        }
        __syncthreads();
    }
    if (col < N && row < M) {
        // scale, convert and clamp
        float scaled = __int2float_rn(acc) * scale_A * scale_B / scale_C;
        acc = min(max(__float2int_rn(scaled) + zero_point_C, -128), 127);
        C[row*N + col] = acc;
    }
}


// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    int T = 32;
    int threadsPerBlock = T*T;
    dim3 numBlocks((N + T - 1) / T, (M + T - 1) / T);
    quant_mm<32><<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);

} 