#include <cuda_runtime.h>
//#include <stdio.h>

__global__ void reverse_array(float* input, int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int T = blockDim.x;
    int base_f = T * bid;
    int base_b = N - (bid + 1) * T;
    // calculate threads position in both ends.
    int f = base_f + tid;
    int b = base_b + (T - 1 - tid);
    int front_valid = min(T, N - base_f);
    int back_valid = min(min(T, N - base_b), T + base_b);
    if (tid < min(front_valid, back_valid)) {
        float x = input[f];
        float y = input[b];
        input[b] = x;
        input[f] = y;

    }
    
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    if (N == 0) return;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 2*threadsPerBlock - 1) / (2 * threadsPerBlock);
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}