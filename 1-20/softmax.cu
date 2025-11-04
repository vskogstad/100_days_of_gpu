#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float smem[512];
    __shared__ float max_x;
    __shared__ float total_sum;
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    float local_max = -INFINITY;

    // Stage 1: find max values (single thread of block)
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        if (input[i] > local_max) local_max = input[i];
    }
    smem[tid] = local_max;
    __syncthreads();
    // Reduce down to one max value in block
    for (int stride = blockDim.x/2; stride >= warpSize; stride/=2) {
        if (tid < stride && smem[tid+stride] > smem[tid]) smem[tid] = smem[tid+stride];
        __syncthreads();
    }
    float val = smem[tid];
    // warp shuffling
    if (tid < warpSize) {
        unsigned mask = __activemask();
        val = fmax(val, __shfl_down_sync(mask, val, 16));
        val = fmax(val, __shfl_down_sync(mask, val, 8));
        val = fmax(val, __shfl_down_sync(mask, val, 4));
        val = fmax(val, __shfl_down_sync(mask, val, 2));
        val = fmax(val, __shfl_down_sync(mask, val, 1));
        if (tid == 0) max_x = val;
    }
    __syncthreads();

    // stage 2: calculate sum
    float local_sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        float e = expf(input[i] - max_x);
        output[i] = e;
        local_sum += e;

    }
    smem[tid] = local_sum;
    __syncthreads();
    // reduce to total sum:
    for (int stride = blockDim.x/2; stride >= warpSize; stride/=2) {
        if (tid < stride) smem[tid] += smem[tid+stride];
        __syncthreads();
    }
    float sm = 0.0f;
    if (tid < warpSize) {
        sm = smem[tid];                  // smem[0..31] are valid partials
        unsigned mask = __activemask();  // warp 0’s lanes
        sm += __shfl_down_sync(mask, sm, 16);
        sm += __shfl_down_sync(mask, sm, 8);
        sm += __shfl_down_sync(mask, sm, 4);
        sm += __shfl_down_sync(mask, sm, 2);
        sm += __shfl_down_sync(mask, sm, 1);
        if ((tid & 31) == 0) total_sum = sm;  // lane 0 writes
    }
    __syncthreads(); // ensure total_sum visible

    //stage 3: Calculate the softmax
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] /= total_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}