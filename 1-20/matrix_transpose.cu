#include <cuda_runtime.h>

template<int T>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // finding the origin of the tile
    int inCol0 = bx * T;
    int inRow0 = by * T;
    int inCol = inCol0 + tx;
    int inRow = inRow0 + ty;
    __shared__ float smem[T][T+1]; // +1 to avoid bank conflicts when loading from smem
    int inIdx = inRow*cols + inCol;
    int outIdx = inCol*rows + inRow; // outRow == inCol, outCol == inRow

    if (inCol < cols && inRow < rows) {
        smem[ty][tx] = input[inIdx];
    }
    __syncthreads();
    // load from columns of smem to rows of output
    if (ty + inCol0 < cols && tx + inRow0 < rows) {
        output[inRow0 + tx + rows*(inCol0 + ty)] = smem[tx][ty];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<32><<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}