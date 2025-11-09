#include <cuda_runtime.h> 

template<int T>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int yi = blockDim.y * blockIdx.y + ty;
    int xi = blockDim.x * blockIdx.x + tx;
    int idx = yi*K + xi;
    float acc = 0.0f;
    // Tile setup
    int numTiles = (N+T-1)/T;
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    for (int t = 0; t < numTiles; t++) {
        // load A and B tiles into smem
        int aCol = t*T + tx;
        int bRow = t*T + ty;
        As[ty][tx] = (yi < M && aCol < N) ? A[N*yi + aCol] : 0.0f;
        Bs[ty][tx] = (xi < K && bRow < N) ? B[K*bRow + xi] : 0.0f;
        __syncthreads();
        // compute partial results for C tile:
        for (int k = 0; k<T; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
        
            
    }
    if (xi < K && yi < M) C[idx] = acc;
        
    }
    
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    int T = 32;
    dim3 threadsPerBlock(T, T);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // Uses a template to pass dimensions to kernel. Need to update both above and below if changing.
    matrix_multiplication_kernel<32><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
