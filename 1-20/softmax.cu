#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float smem[512];
    __shared__ float max_mem[512];
    __shared__ float max_x;
    __shared__ float total_sum;
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    float local_max = -INFINITY;

    // Stage 1: find max values (single thread of block)
    float m_t = -INFINITY;
    float s_t = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        float x = input[i];
        if (x > m_t) {
            s_t = s_t * expf(m_t-x) + 1;  // 1 = expf(x-x)
            m_t = x;
        }
        else s_t += expf(x - m_t);
    }
    smem[tid] = s_t;
    max_mem[tid] = m_t;
    __syncthreads();
    
    // Reduce down to one max value and one sum in block
    for (int stride = blockDim.x/2; stride >= 1; stride/=2) {
        if (tid < stride && tid + stride < N) {
            if (max_mem[tid] < max_mem[tid+stride]) {
                // m2 > m1
                smem[tid] = smem[tid] * expf(max_mem[tid] - max_mem[tid+stride]) + smem[tid+stride];
                max_mem[tid] = max_mem[tid+stride];
            }
            else {
                // m1 > m2
                smem[tid] = smem[tid] + expf(max_mem[tid+stride] - max_mem[tid]) * smem[tid+stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0){
        max_x = max_mem[0];
        total_sum = smem[0];
    }
    __syncthreads();
    /*float val = smem[tid];
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
    __syncthreads();*/

    //stage 2: Calculate the softmax
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = expf(input[i] - max_x) / total_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}