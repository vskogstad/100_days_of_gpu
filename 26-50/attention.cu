#include <cuda_runtime.h>

#template<t>
__global__ void softmax_attention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // The Q axis is grouped. It wil be partinioned into M_mem/N*D different blocks.
    // softmax(Q K^T) / sqrt(d) Each row of softmax is independent so we can split it across numRow blocks
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    float scale = __fsqrt_rn(d);
    // do tiled matrix multiply
    int numTiles = (t + N) / t;
    __shared__ Q_smem[t][t];
    __shared__ K_smem[t][t];
    for (int tileIdx = ?; tileIdx < numTiles; tileIdx++) {

    }
    __syncthreads();
    

    // multiply by V

}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    t = 32;
    threadsPerBlock = 1024;
    numBlocks = M; //(threadsPerBlock + N) / threadsPerBlock;
    softmax_attention<32><<NumBlocks, threadsPerBlock>>(Q, K, V, output, M, N, d);

}
