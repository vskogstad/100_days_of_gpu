%%writefile GELU.cu

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>



#define CHECK(call) do {                                           \
  cudaError_t err = (call);                                        \
  if (err != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
            cudaGetErrorString(err), __FILE__, __LINE__);          \
    return 1;                                                      \
  }                                                                \
} while (0)



__global__ void GELU_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        output[i] = 0.5f * input[i] * (1 + tanhf(sqrtf(0.79788456f) * (input[i] + 0.044715f * ( input[i] * input[i] * input[i]))));
        // 2.0f/M_PI = 0.79788456f
        // Relu:                      output[i] = fmaxf(input[i], 0.0f);
        // Leaky ReLU (alpha 0.01):   output[i] = fmaxf(input[i], 0.01f*input[i]);
        // Hardtanh (-1, 1):        ouutput[i] =fmin(fmaxf(input[i], -1.0f), 1.0f);
    }

}

// input, output are device pointers
int main() {
    int N = 1024;
    float input[N], output[N];
    float *d_input, *d_output;

    // create dummy values
    for (int i = 0; i < N; i++) {
        input[i] = i;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // allocate space on device and transfer input to device
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    // launch kernel
    GELU_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    // print output
    for (int i = 0; i < 5; i++) {
        printf("%f\n", output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}


