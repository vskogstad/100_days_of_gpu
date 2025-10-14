#include <cuda_runtime.h>
// This kernel adds two vectors A and B of size N and stores the result in vector C.
// Based on LeetGPU scaffolding. I added memory management and kernel.
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }

}

int main() {
    int N = 12;
    float h_A[N], h_B[N], h_C[N]; // host vectors
    float *d_A, *d_B, *d_C; // device vectors
    // crete test data
    for (int i = 0; i < N; i++) {
      h_A[i] = i;
      h_B[i] = 2*i;
    }
    for (int i = 0; i < N; i++) {
      printf("%f ", h_A[i]);
    }
    printf("\n");

    int blockDim = 4; // Number of threads per block
    int gridDim = (N + blockDim - 1) / blockDim; // Number of blocks
    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    cudaMalloc((void**)&d_B, N * sizeof(float));    
    cudaMalloc((void**)&d_C, N * sizeof(float));
    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    // Launch kernel
    vector_add<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    // Copy result vector from device to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  
    // Print result
    for (int i = 0; i < N; i++) {
      printf("%f ", h_C[i]);
    }
    return 0;
}