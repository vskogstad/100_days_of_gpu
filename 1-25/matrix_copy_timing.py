import cupy as cp

src = r"""#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, const int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = N*N;
    const int full = total/4;
    const int start_tail = 4*full;
    const int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < full; i+=stride) {
        // cast to float4
        
        reinterpret_cast<float4*>(B)[i] = reinterpret_cast<const float4*>(A)[i];
        //B[i] = A[i];

    }
    
    if (idx == 0) {
        for (int j = start_tail; j < total; j++) {
            B[j] = A[j];
        }
    }
}
"""
T = 1024 # or 32
mod = cp.RawModule(code=src,
                   options=('--std=c++11', '-Xptxas', '-v', '-maxrregcount=16'),
                   name_expressions=[f'copy_matrix_kernel'])
ker = mod.get_function(f"copy_matrix_kernel")
# Printing
attrs = ker.attributes  # CUDA function attributes
print("num_regs:", attrs['num_regs'])
print("shared_size_bytes:", attrs['shared_size_bytes'])
print("max_threads_per_block:", attrs['max_threads_per_block'])
print("local_size_bytes:", attrs['local_size_bytes'])
print("ptx_version:", attrs.get('ptx_version'))
print("binary_version:", attrs.get('binary_version'))


def time_kernel(N=1024, T=256, blocks=1280):
    M, N = 8192, 8192

    A = cp.random.rand(M, N, dtype=cp.float32)
    B = cp.empty((M, N), dtype=cp.float32)

    # warmup
    for _ in range(130):
      ker(((M*N+T-1)//T, 1, 1), (T, 1, 1), (A, B, N))
    cp.cuda.runtime.deviceSynchronize()

    # measure
    start = cp.cuda.Event(); end = cp.cuda.Event()
    iters = 150
    F = M*N//4
    start.record()
    for _ in range(iters):
        ker(((F+T-1)//T, 1, 1), (T, 1, 1), (A, B, N))
        #ker((blocks,), (T,), (x, blockSum, N), shared_mem=shmem)
    end.record(); end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end) / iters

    # GB/s (each element read once)
    gbs = (2*(M * N) * A.itemsize) / (ms * 1e-3) / 1e9
    #GFLOP_per_s = 2 * M * N * K / (ms * 1e-3) / 1e9
    return ms, gbs

ms, gbs= time_kernel(N=1024, T=T)
print(f"{ms:.3f} ms/iter, {gbs:.1f} GB/s")


