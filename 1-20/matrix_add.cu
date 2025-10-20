#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int M, int N, int K) {
    int yi = blockDim.y * blockIdx.y + threadIdx.y;
    int xi = blockDim.x * blockIdx.x + threadIdx.x;
    if (xi < N && yi < M) {
        int idx = yi*N + xi;
        C[idx] = A[idx] + B[idx];
    }
    
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
