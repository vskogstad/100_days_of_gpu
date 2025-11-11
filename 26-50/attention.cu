#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void softmax_attention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // The Q axis is grouped. It wil be partinioned into M_mem/d different blocks.
    // softmax(Q K^T) / sqrt(d) Each row of softmax is independent so we can split it across numRow blocks, does it make sense though?

    int tid = threadIdx.x;
    int row_idx = blockIdx.x * d;
    float scale = __fsqrt_rn(d);
    __shared__ float Q_s[1024];
    __shared__ float s_sum[1024];
    __shared__ float s_max[1024];
    __shared__ float max_x;
    __shared__ float total_sum;
    __shared__ float acc[1024];
    // read Q into shared memory
    for (int i = tid; i < d; i+=blockDim.x){
            acc[i] = 0.0f;
        }
    __syncthreads();


    // read Q into shared memory
    for (int i = tid; i < d; i+=blockDim.x){
            Q_s[i] = Q[i + row_idx];
        }
    
    __syncthreads();

    // loop over all N values of K and V
    float x_max = -INFINITY;
    float s = 0.0f;
    
    // iterate over K^T
    float x_new;
    bool first = true;
    for (int j = tid; j < N; j+=blockDim.x) {
        // threadwise max_x and sum
        float dot = 0.0f;
        for (int t = 0; t < d; t++){
            dot +=  Q_s[t] * K[j*d + t];
        }
        x_new = dot / scale;
        if (first) {
            s = 1; 
            x_max = x_new; 
            first = false;
        }
        else if (x_new > x_max) {
            s = s * __expf(x_max-x_new) + 1;
            x_max = x_new;
        }
        else {
            s = s + __expf(x_new-x_max);
        }
    }
    s_sum[tid] = s;
    //printf("tid:%d => s:%f | x = %f\n", tid, s, x_max);
    s_max[tid] = x_max;
    __syncthreads();

    // reduce to single softmax sum and max
    int grid_size = min(N+1, blockDim.x);
    //printf("T = %d\n", T);
    for (int stride = grid_size/2; stride >=1; stride /= 2) {
        if ((stride + tid) < grid_size && tid < stride) {
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
    // Pass 2:
    // loop over all threads and compute w = exp(sj-x_max)/sum.
    //  multiply by V right away and directly update output vector using atomic add

    // tile by 32 to avoid contention during atomicAdd
    const int T = 32;
    for (int t0 = 0; t0 < d; t0+=T) {
        float local_acc[T];
        // zero registers
        for (int u = 0; u < T; u++) local_acc[u] = 0.0f;

        for (int j = tid; j < N; j+=blockDim.x) {
            float dot = 0.0f;
            for (int t = 0; t < d; t++){
                dot +=  Q_s[t] * K[j*d + t];
                }
            x_new = dot / scale;
            float w = __expf(x_new - max_x) / total_sum;
            for (int u = 0; u < T; u++){
                int t = t0 + u;
                if (t < d) local_acc[u] += w * V[j*d + t];
            }
        }

        for (int u = 0; u < T; u++) {
            int t = t0 + u;
            if (t < d) atomicAdd(&output[row_idx + t], local_acc[u]);
        }
    }

}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {

    int threadsPerBlock = 32;
    int numBlocks = M; //(threadsPerBlock + N) / threadsPerBlock;
    softmax_attention<<<numBlocks, threadsPerBlock>>>(Q, K, V, output, M, N, d);

}
