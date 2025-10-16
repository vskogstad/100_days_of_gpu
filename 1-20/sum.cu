#include <cuda_runtime.h>

// kernel for summing all threads in block
__global__ void reduce(const float* input, float* blockSum, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x * 2 + tid;

    // First sum, get values into shared memory
    float sum = 0.0f;
    if (i < N) {
        sum += input[i];
        if (i + blockDim.x < N) {
            sum += input[i+blockDim.x];
        } 
    }
    smem[tid] = sum;
    __syncthreads();
    // For loop reducing the size of the stride each iteration
    // Alt (int stride = blockdim.x >> 1; stride > 0 stride >>= 1)
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
       
    // write result to blockSum
    if (tid == 0) {
        blockSum[blockIdx.x] = smem[0];
    }

    

}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 256;
    int numBlocks = (threadsPerBlock*2 + N - 1) / (threadsPerBlock * 2); 
    size_t sharedBytes = threadsPerBlock * sizeof(float);

    float* blockSum = nullptr;
    cudaMalloc(&blockSum, numBlocks*sizeof(float));
    float* h_blockSum = (float*)malloc(numBlocks * sizeof(float));
    //float h_blockSum[numBlocks];
    // first reduce to one result per Block
    reduce<<<numBlocks, threadsPerBlock, sharedBytes>>>(input, blockSum, N);
    // sum up the resulting sum in each block on CPU
    cudaMemcpy(h_blockSum, blockSum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    double acc = 0.0;
    for (int i=0; i<numBlocks; i++) {
        acc += (double)h_blockSum[i];
    }
    float result = (float)acc;

    cudaMemcpy(output, &result, sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(blockSum);
    free(h_blockSum);
    
   


}


