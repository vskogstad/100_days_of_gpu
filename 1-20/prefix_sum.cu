#include <cuda_runtime.h>

__global__ void prefix_sum(const float* input, float* output, float* blockSums, float* blockOffsets, int N) {
    __shared__ float smem[512];
    int t = threadIdx.x;
    int blockStart = blockDim.x * blockIdx.x;
    int i = blockStart + t;
    int n_active = max(0, min(N - blockStart, 512));
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
    // update block sum and zero out.
    if (t == limit-1) {
        if (blockSums) blockSums[blockIdx.x] = smem[limit-1]; // only if provided.
        smem[limit-1] = 0.0f;
    }
    __syncthreads();
    // sweep down
    for (int stride = limit/2; stride >= 1; stride /= 2) {
        int j = (t + 1) * 2 * stride - 1; 
        if (j < limit) {
            float temp = smem[j-stride];
            smem[j-stride] = smem[j];
            smem[j] += temp;
            
        }
        __syncthreads();
    }
    if (i < N) {
        output[i] = smem[t];
        if (blockSums) output[i] += input[i];  // Inclusive prefixSum (exclusive for calc. offsets)
        if (blockOffsets) output[i] += blockOffsets[blockIdx.x];    
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 512;
    int numBlocks = (threadsPerBlock + N - 1)/ threadsPerBlock;
    // Adding block sum and offset to accumulate across blocks
    float* blockSums = nullptr;
    float* blockOffsets = nullptr;
    cudaMalloc(&blockSums, sizeof(float)*numBlocks);
    cudaMalloc(&blockOffsets, sizeof(float)*numBlocks);
    // Three passes of same kernel:
    // -First pass (input) gives prefixes (inclusive) within each block. We get the total blockSum for each block.
    // -Second pass (blockSums) gives us prefixes (exclusive) across blockSums. We us this to update blockOffsets.
    // -Third pass, (input) but now we add the blockOffsets, which gives us our final answer.
    prefix_sum<<<numBlocks, threadsPerBlock>>>(input, output, blockSums, nullptr, N); // prefix within each block
    prefix_sum<<<numBlocks, threadsPerBlock>>>(blockSums, blockOffsets, nullptr, nullptr, numBlocks); //
    prefix_sum<<<numBlocks, threadsPerBlock>>>(input, output, blockSums, blockOffsets, N);


} 
