#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float acc = 0.0f;
    __shared__ float smem[2048];
    for (int i=threadIdx.x; i <kernel_size; i+= blockDim.x) {
        smem[i] = kernel[i];
    }
    __syncthreads();    
    if (idx < input_size-kernel_size+1) {
        #pragma unroll
        for (int j = 0; j < kernel_size; j++) {
            acc += input[idx + j] * smem[j];
        }
        output[idx] = acc;
        
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}