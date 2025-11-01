#include <cuda_runtime.h>

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
__global__ void final_count(int* output, int N, int K, int* blockSum) {
    __shared__ int smem[512];    
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int sum = 0;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) 
        smem[i] = 0;
        __syncthreads();
    //Register number of matches within each thread
    for (int i = idx; i < blockDim.x; i+= blockDim.x * gridDim.x) {
        sum += blockSum[i];
    }
    smem[tid] = sum;
    __syncthreads();
    
    // Reduce from individual threads down to total count
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (tid < stride) smem[tid] += smem[tid+stride];
        __syncthreads();
    }
    
    // Write total count to output pointer
    if (tid == 0) {
        *output = smem[0]; 
    }
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
    final_count<<<1, blocksPerGrid>>>(output, N, K, blockSum_d);
    cudaDeviceSynchronize();
    cudaGetLastError();



}