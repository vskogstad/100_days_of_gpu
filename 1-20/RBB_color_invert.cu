#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    const int idx = (blockDim.x * blockIdx.x + threadIdx.x);
    uchar4* __restrict__ pix = reinterpret_cast<uchar4*>(image);
    const int limit = width*height;
    const int stride = blockDim.x*gridDim.x;
    // grid stide (with two pixels per load)
    for (int i = idx; i < limit; i += 2*stride) {
        uchar4 v = pix[i];
        v.x = 255-v.x;
        v.y = 255-v.y;
        v.z = 255-v.z;
        pix[i] = v;
        // loading 2nd pixel 
        if (i+stride < limit) {
            uchar4 v2 = pix[i+stride];
            v2.x = 255-v2.x;
            v2.y = 255-v2.y;
            v2.z = 255-v2.z;
            pix[i+stride] = v2;
        }



    }

}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}