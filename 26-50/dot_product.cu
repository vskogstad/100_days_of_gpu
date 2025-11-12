#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    /// Thinking we could do this naively first then split into warp shuffle blocks
    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    //__shared__ float sum;
    float acc = 0.0f;
    for (int i = idx; i < N; i += blockDim.x*gridDim.x) {
        //float sum = 0.0f;
        acc += A[i] * B[i];
        //printf("i: %i, p: %f\n", i, acc);
        // warp shuffle into atomicAdd first then smem later
        if (tid < warpSize) {
            
            unsigned mask = __activemask();
            float p = acc;
            for (int stride = warpSize / 2; stride >= 1; stride /= 2) {
                p += __shfl_down_sync(mask, p, stride);
                //printf("Stride: %i, p: %f\n", stride, p);

            }
        
        if (tid == 0) {
            //printf("Hi %f\n", p);
            atomicAdd(&result[0],p);
        }}


    }

} 


// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 32;
    int numBlocks = (threadsPerBlock + N - 1) / threadsPerBlock;
    dot_product<<<numBlocks, threadsPerBlock>>>(A, B, result, N);
}