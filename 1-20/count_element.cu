#include <cuda_runtime.h>
#include <stdio.h>
#include <numeric>
#include <vector>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K, int* blockSum) {
    __shared__ int blockCount[512];    
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int sum = 0;
    //Register number of matches within each thread
    for (int i = idx; i < N; i+= blockDim.x * gridDim.x) {
        if (input[i] == K)  sum += 1;
    }
    blockCount[tid] = sum;
    __syncthreads();
    // Reduce from individual threads down to block count
    
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (tid < stride) blockCount[tid] += blockCount[tid+stride];
        __syncthreads();
    }
    // Write blockCount
    if (tid == 0) {
        blockSum[blockIdx.x] = blockCount[0];
    }
        
}


// Sum and return results
__global__ void final_count(int* output, int numBlocks, const int* blockSum) {
    __shared__ int smem[1024];          // must be >= TB
    int tid = threadIdx.x;

    int local = 0;
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        local += blockSum[i];
    }
    smem[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *output = smem[0];
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int* blockSum_d = nullptr;
    cudaMalloc(&blockSum_d, blocksPerGrid*sizeof(int));

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K, blockSum_d);
    cudaDeviceSynchronize();
    cudaGetLastError();
    std::vector<int> h(blocksPerGrid);
    cudaMemcpy(h.data(), blockSum_d, blocksPerGrid*sizeof(int), cudaMemcpyDeviceToHost);
    int host_sum_blocks = std::accumulate(h.begin(), h.end(), 0);
    //cudaMemcpy(output, &host_sum_blocks, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("___________ %i", host_sum_blocks); //}/*
    int TB = 1; while (TB < blocksPerGrid && TB < 1024) TB <<= 1;
    final_count<<<1, TB>>>(output, blocksPerGrid, blockSum_d);
    cudaDeviceSynchronize();
    cudaGetLastError();
    
    }//*/



