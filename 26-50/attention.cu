#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_attention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // The Q axis is grouped. It wil be partinioned into M_mem/d different blocks.
    // softmax(Q K^T) / sqrt(d) Each row of softmax is independent so we can split it across numRow blocks, does it make sense though?

    int tid = threadIdx.x;
    int base = blockDim.x * blockIdx.x;
    int idx = base + tid;
    float scale = __fsqrt_rn(d);
    __shared__ float Q_s[1024];
    __shared__ float s_sum[1024];
    __shared__ float s_max[1024];
    __shared__ float max_x;
    __shared__ float total_sum;
    // read Q into shared memory
    for (int i = tid; i < d; i+=blockDim.x){
            Q_s[i] = Q[i + base];
        }
    
    __syncthreads();

    // loop over all N values of K and V
    float x_max = -INFINITY;
    float s = 0.0f;
    
    // threadwise max_x and sum
    if (tid < d){
        float x_new;
        for (int j = 0; j < N; j+=blockDim.x) {
            x_new = Q_s[tid] * K[j] / scale;
            if (x_new > x_max) {
                s = s * __expf(x_new-x_max) + 1;
                x_max = x_new;}
            else {
                s = s + __expf(x_new-x_max);
            }
        }
        s_sum[tid] = s;
        s_max[tid] = x_max;
        __syncthreads();
        // reduce to single softmax sum and max
        for (int stride = blockDim.x/2; stride >=1; stride /= 2) {
            if (tid > stride) {
                float s1 = s_sum[tid];        float s2 = s_sum[tid+stride];
                float x1 = s_max[tid];        float x2 = s_max[tid+stride];

                if (s2 == 0.0f)         {/*Do nothing*/}
                else if (s1 == 0.0f)    {s_sum[tid] = s2; s_max[tid] = x2;}
                else if (x1 < x2)       {s_sum[tid] = s1 * __expf(x1-x2) + s2;    s_max[tid] = x2;}
                else                    {s_sum[tid] = s1 + s2 * __expf(x2-x1);}
            }
            __syncthreads();
        }
        if (tid == 0) {max_x = s_max[0]; total_sum = s_sum[0];}
        __syncthreads();
        // loop over all threads and compute w = exp(sj-x_max)/sum.
        //  multiply by V right away and directly update output vector using atomic add

        for (int j = 0; j < N; j+=blockDim.x) {
            x_new = Q_s[tid] * K[j] / scale;
            s = __expf(x_new - max_x) / total_sum;
            atomicAdd(&output[j], s * V[j]);
        }

    //
    }

}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {

    int threadsPerBlock = 1024;
    int numBlocks = M; //(threadsPerBlock + N) / threadsPerBlock;
    softmax_attention<<<numBlocks, threadsPerBlock>>>(Q, K, V, output, M, N, d);

}
