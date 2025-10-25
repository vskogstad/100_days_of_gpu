#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    const int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    float acc = 0.0f;
    int smem_size = 512 + 2048;
    int output_size = input_size-kernel_size+1;
    int num_tiles = output_size/512;
    // Tiled calculation
    for (int t = 0; t < num_tiles; t++) {
        // load input relevant to tile into smem
        __shared__ float smem[2550];
        for (int i=0; i <smem_size; i+=blockDim.x) {
            if (idx+i < input_size) smem[tid + i] = input[idx+i];
        }
        __syncthreads();  
        // convolution  
        if (idx < input_size-kernel_size+1) {
            #pragma unroll
            for (int j = 0; j < kernel_size; j++) {
                acc += smem[tid + j] * kernel[j];
            }
            output[idx] = acc;
            
        }
    }

    }



// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 512;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}