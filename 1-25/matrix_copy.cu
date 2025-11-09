#include <cuda_runtime.h>
// The naive matrix copy kernel
__global__ void copy_matrix_kernel(const float* __restrict__  A, float* B, const int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N*N;
    if (idx < total) {
        B[idx] = A[idx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 512;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
} 