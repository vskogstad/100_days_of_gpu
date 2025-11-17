#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    // Thinking we could load and add directly to output
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int stride = blockDim.x * gridDim.x;
    int limit = N*N;
    for (int i = idx; i < limit; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}