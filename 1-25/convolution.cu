#include <cuda_runtime.h>
__constant__ float Kc[2048]; // assuming max kernel size is 2048

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    extern __shared__ float smem[];
    
    const int tid = threadIdx.x;
    const int base = blockDim.x * blockIdx.x;
    const int idx = base + tid;
    const int T = blockDim.x + kernel_size - 1;
    int output_size = input_size-kernel_size+1;
    // coop load input relevant to block into smem
    for (int i=tid; i < T; i+=blockDim.x) {
        if (base+i < input_size) smem[i] = input[base+i];
    }
    __syncthreads();  
    // convolution  
    if (idx < output_size) {
        float acc = 0.0f;
        for (int j = 0; j < kernel_size; j++) {
            acc += smem[tid + j] * Kc[j];
        }
        output[idx] = acc;
        
    }
}


// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 512;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    size_t shmemBytes = (512 + kernel_size - 1) * sizeof(float); // dynamic shared memory size
    
    cudaMemcpyToSymbol(Kc, kernel, kernel_size * sizeof(float));

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, shmemBytes>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}