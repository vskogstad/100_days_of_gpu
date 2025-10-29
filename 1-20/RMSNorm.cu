#include <cuda_runtime.h>
//#include <math.h>

__global__ void RMSNorm(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    __shared__ float acc[256];
    if (i == 0) acc[0] = 0.0f;
    __syncthreads();
    
    float x_pow = (i < N) ? input[i]*input[i] : 0.0f;
    if (i < N) atomicAdd(&acc[0], x_pow);
    __syncthreads();
    if (i == 0) {
        acc[0] = sqrtf((acc[0]/N) + eps);
    }
    __syncthreads();
    if (i < N) {
        float x_hat = input[i] / acc[0];
        output[i] = gamma * x_hat + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
                        int threadsPerBlock = 256;
                        int blocksPerGrid = (threadsPerBlock + N - 1) / threadsPerBlock;
                        RMSNorm<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, eps);

}
