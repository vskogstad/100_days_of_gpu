#include <cuda_runtime.h>

__global__ void prefix_sum(const float* input, float* output, int N) {
    __shared__ float smem[256];
    int t = threadIdx.x;
    int blockStart = blockDim.x * blockIdx.x;
    int i = blockStart + t;
    int n_active = max(0, min(N - blockStart, 256));
    int limit = 1; while (limit < n_active) limit <<= 1;
    if (i < N) smem[t] = input[i];
    else smem[t] = 0;
    __syncthreads();
    // sweep up
    for (int stride = 1; stride <= limit/2; stride*=2) {
        int j = (t + 1) * 2 * stride - 1;
        if (j < limit) smem[j] += smem[j-stride];
        __syncthreads();
    }
    // sweep down
    if (t == limit-1) smem[limit-1] = 0;
    __syncthreads();
    for (int stride = limit/2; stride >= 1; stride /= 2) {
        int j = (t + 1) * 2 * stride - 1; 
        if (j < limit) {
            float temp = smem[j-stride];
            smem[j-stride] = smem[j];
            smem[j] += temp;
            
        }
        __syncthreads();
    }
    if (i < N) output[i] = smem[t] + input[i];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    
    int threadsPerBlock = 256;
    int numBlocks = (threadsPerBlock + N - 1)/ threadsPerBlock;

    prefix_sum<<<numBlocks, threadsPerBlock>>>(input, output, N);
} 
