#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float smem[512];
    __shared__ float max_x;
    __shared__ float total_sum;
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    float local_max = -INFINITY;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        if (input[i] > local_max) local_max = input[i];
    }
    smem[tid] = local_max;
    __syncthreads();
    // Reduce down to one max value
    for (int stride = blockDim.x/2; stride > 0; stride/=2) {
        if (tid + stride < N && smem[tid+stride] > smem[tid]) smem[tid] = smem[tid+stride];
        __syncthreads();
    // warp shuffling here later
    }
    max_x = smem[0]; // smem[0] Now contains the maximum

    // stage 2: calculate sum
    float local_sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += expf(input[idx] - max_x);
    }
    smem[tid] = local_sum;
    // reduce to total sum:
    for (int stride = blockDim.x/2; stride > 0; stride/=2) {
        if (tid + stride < N) smem[tid] += smem[tid+stride];
        __syncthreads();
    }
    total_sum = smem[0]; //smem[0] containing sum

    //stage 3: Calculate the softmax
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = expf(input[i] - max_x)/total_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}