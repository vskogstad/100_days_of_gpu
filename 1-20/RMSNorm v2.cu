#include <cuda_runtime.h>
//#include <math.h>

__global__ void PowerSum(const float* input, int N, float* blockSum) {
    int tid = threadIdx.x;
    int idx = tid + (blockDim.x * blockIdx.x);
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
    for (int stride = blockDim.x/2; stride >= warpSize; stride /= 2) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
        }
    // warp shuffle 

    if (tid < warpSize) {
        float val = acc[tid];
        unsigned mask = __activemask();
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
        if (tid == 0) blockSum[blockIdx.x] = val;
    }

    // ensure single block has all accumulated sums:

    
    /*
    __all_sync();
    // reduce down to just one value
    if (idx == 0) {
        acc[0] = 0.0f;
        for (int j = idx; idx < gridDim.x; j++) {
            acc[0] += blockSum[idx];
        }
        blockSum[0] = sqrtf((acc[0]/N) + eps);
    }
    */
}
__global__ void RMSNorm(const float* input, float gamma, float beta, float* output, int N, float eps, float rmsn) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    for (int k = idx; k < N; k+= blockDim.x*gridDim.x) {
        float x_hat = input[k] / rmsn;
        output[k] = gamma * x_hat + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
                        int threadsPerBlock = 256;
                        int blocksPerGrid = (threadsPerBlock + N - 1) / threadsPerBlock;
                        float* blockSum_d = nullptr;
                        cudaMalloc(&blockSum_d, blocksPerGrid*sizeof(float));
                        PowerSum<<<blocksPerGrid, threadsPerBlock>>>(input, N, blockSum_d);
                        cudaDeviceSynchronize();

                        std::vector<float> blockSum_h(blocksPerGrid);
                        cudaMemcpy(blockSum_h.data(), blockSum_d, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
                        float rmsn = 0.0f;
                        for (int j = 0; j < blocksPerGrid; j++) {
                            rmsn += blockSum_h[j];
                        }
                        rmsn = sqrtf((rmsn/N) + eps);
                        RMSNorm<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, eps, rmsn);

}