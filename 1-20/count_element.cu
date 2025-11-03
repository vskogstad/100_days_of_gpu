#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
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
    
    for (int stride = blockDim.x/2; stride >= warpSize; stride /= 2) {
        if (tid < stride) blockCount[tid] += blockCount[tid+stride];
        __syncthreads();
    }
    int val = blockCount[tid];
    if (tid < warpSize) {
        unsigned mask = __activemask();
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
    }
    
    // Write to output
    if (tid == 0) {
        atomicAdd(output, val);
    }
        
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
    cudaGetLastError();
}




