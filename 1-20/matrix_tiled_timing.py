import cupy as cp

src = r"""
#include <cuda_runtime.h> 

template<int T>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
    
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
"""
T = 32  # or 32
mod = cp.RawModule(
    code=src,
    options=("--std=c++11", "-Xptxas", "-v"),
    name_expressions=[f"matrix_multiplication_kernel<{T}>"],
)
ker = mod.get_function(f"matrix_multiplication_kernel<{T}>")
# Printing
attrs = ker.attributes  # CUDA function attributes
print("num_regs:", attrs["num_regs"])
print("shared_size_bytes:", attrs["shared_size_bytes"])
print("max_threads_per_block:", attrs["max_threads_per_block"])
print("local_size_bytes:", attrs["local_size_bytes"])
print("ptx_version:", attrs.get("ptx_version"))
print("binary_version:", attrs.get("binary_version"))


def time_kernel(N=1024, T=16, blocks=1280):
    M, N, K = 2048, 2048, 2048

    A = cp.random.rand(M, N, dtype=cp.float32)
    B = cp.random.rand(N, K, dtype=cp.float32)
    C = cp.empty((M, K), dtype=cp.float32)

    # warmup
    for _ in range(30):
        ker(((K + T - 1) // T, (M + T - 1) // T, 1), (T, T, 1), (A, B, C, M, N, K))
    cp.cuda.runtime.deviceSynchronize()

    # measure
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    iters = 50
    start.record()
    for _ in range(iters):
        ker(((K + T - 1) // T, (M + T - 1) // T, 1), (T, T, 1), (A, B, C, M, N, K))
        # ker((blocks,), (T,), (x, blockSum, N), shared_mem=shmem)
    end.record()
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end) / iters

    # GB/s (each element read once)
    gbs = ((M * N + N * K + K * M) * A.itemsize) / (ms * 1e-3) / 1e9
    GFLOP_per_s = 2 * M * N * K / (ms * 1e-3) / 1e9
    return ms, gbs, GFLOP_per_s


ms, gbs, gfs = time_kernel(N=1024, T=T)
print(f"{ms:.3f} ms/iter, {gbs:.1f} GB/s {gfs:.1f} GFLOP/s")
