#include <cuda_runtime.h>
//#include <math.h>

__global__ void RMSNorm(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    __shared__ float acc[256];
    float x_pow = 0.0f;
    if (idx == 0) acc[0] = 0.0f;
    __syncthreads();
    for (int i = idx; i < N; i+= blockDim.x*gridDim.x) {
        float x = input[i];
        x_pow += x * x;
    }
    acc[threadIdx.x] = x_pow;
    __syncthreads();
    // Using reduction to accumulate sum into acc[0]
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (idx < stride) acc[idx] += acc[idx+stride];
        __syncthreads();
        }
    // can add warp shuffle here later


    // ensure single block has all accumulated sums:

    if (tid == 0) blocksum[blockIdx.x] = acc[0];
    __all_sync();
    // reduce down to just one value
    if (blockIdx.x == 0) {
        (idx < gridDim.x) ? acc[idx] = blocksum[idx]: acc[idx] = 0.0f;
    }
    __syncthreads()
    if (idx == 0) {

        acc[0] = sqrtf((acc[0]/N) + eps);
    }
    __syncthreads();
    for (int k = idx; k < N; k+= blockDim.x*gridDim.x) {
        float x_hat = input[k] / acc[0];
        output[k] = gamma * x_hat + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
                        int threadsPerBlock = 256;
                        int blocksPerGrid = (threadsPerBlock + N - 1) / threadsPerBlock;
                        RMSNorm<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, eps);

}
