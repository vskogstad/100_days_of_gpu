from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cpp_src = r"""
// Forward declaration so the auto-generated pybind code sees it. Needed for code to compile.
void nvfp4_gemv(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
);
"""

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define CEIL_DIV(A, B) ((A + B - 1) / B)

// Tile size in K_blocks (each K_block = 8 FP4x2 pairs = 16 FP4 values)
#define TILE_KB 128
#define PAIRS_PER_BLOCK 8

__global__ void gemv(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ sfa,
    const uint8_t* __restrict__ sfb,
    at::Half* __restrict__ C,
    int M, int K_pairs, int K_blocks, int L,
    int a_s0, int a_s1, int a_s2,
    int b_s1, int b_s2,
    int sfa_s0, int sfa_s1, int sfa_s2,
    int sfb_s1, int sfb_s2
) {
    // Shared memory for B tile and scale factors
    __shared__ __nv_fp4x2_storage_t B_tile[TILE_KB * PAIRS_PER_BLOCK];  // 512 bytes
    __shared__ __nv_fp8_storage_t sfb_tile[TILE_KB];                     // 64 bytes

    auto* a_fp4x2 = reinterpret_cast<const __nv_fp4x2_storage_t*>(A);
    auto* b_fp4x2 = reinterpret_cast<const __nv_fp4x2_storage_t*>(B);
    auto* Sa_fp8 = reinterpret_cast<const __nv_fp8_storage_t*>(sfa);
    auto* Sb_fp8 = reinterpret_cast<const __nv_fp8_storage_t*>(sfb);

    int tid = threadIdx.x;
    int m = blockIdx.x * blockDim.x + tid;
    int l = blockIdx.z;

    float acc = 0.0f;

    // Tile over K_blocks dimension
    for (int kb_base = 0; kb_base < K_blocks; kb_base += TILE_KB) {
        int tile_kb = min(TILE_KB, K_blocks - kb_base);
        int tile_pairs = tile_kb * PAIRS_PER_BLOCK;

        // === Cooperative load of B into shared memory ===
        for (int i = tid; i < tile_pairs; i += blockDim.x) {
            int local_kb = i / PAIRS_PER_BLOCK;
            int local_t = i % PAIRS_PER_BLOCK;
            int global_pair = (kb_base + local_kb) * PAIRS_PER_BLOCK + local_t;
            
            int64_t B_idx = global_pair * b_s1 + l * b_s2;
            B_tile[i] = b_fp4x2[B_idx];
        }

        // === Cooperative load of sfb into shared memory ===
        for (int i = tid; i < tile_kb; i += blockDim.x) {
            int64_t sfb_idx = (kb_base + i) * sfb_s1 + l * sfb_s2;
            sfb_tile[i] = Sb_fp8[sfb_idx];
        }

        __syncthreads();

        // === Process tile ===
        if (m < M) {
            for (int local_kb = 0; local_kb < tile_kb; local_kb++) {
                int global_kb = kb_base + local_kb;

                // Load sfa from global (each thread needs different row)
                int64_t sfa_idx = m * sfa_s0 + global_kb * sfa_s1 + l * sfa_s2;
                __half_raw sa_raw = __nv_cvt_fp8_to_halfraw(Sa_fp8[sfa_idx], __NV_E4M3);
                __half sa = *reinterpret_cast<__half*>(&sa_raw);

                // Load sfb from shared (same for all threads)
                __half_raw sb_raw = __nv_cvt_fp8_to_halfraw(sfb_tile[local_kb], __NV_E4M3);
                __half sb = *reinterpret_cast<__half*>(&sb_raw);

                // Inner loop over pairs in this block
                #pragma unroll
                for (int t = 0; t < PAIRS_PER_BLOCK; t++) {
                    int global_pair = global_kb * PAIRS_PER_BLOCK + t;
                    int tile_idx = local_kb * PAIRS_PER_BLOCK + t;

                    // Load A from global
                    int64_t A_idx = m * a_s0 + global_pair * a_s1 + l * a_s2;
                    __nv_fp4x2_storage_t a_pair = a_fp4x2[A_idx];

                    // Load B from shared
                    __nv_fp4x2_storage_t b_pair = B_tile[tile_idx];

                    // Convert FP4 -> half2
                    __half2_raw h2a_raw = __nv_cvt_fp4x2_to_halfraw2(a_pair, __NV_E2M1);
                    __half2 h2a = *reinterpret_cast<__half2*>(&h2a_raw);
                    __half a0 = __hmul(__low2half(h2a), sa);
                    __half a1 = __hmul(__high2half(h2a), sa);

                    __half2_raw h2b_raw = __nv_cvt_fp4x2_to_halfraw2(b_pair, __NV_E2M1);
                    __half2 h2b = *reinterpret_cast<__half2*>(&h2b_raw);
                    __half b0 = __hmul(__low2half(h2b), sb);
                    __half b1 = __hmul(__high2half(h2b), sb);

                    acc += __half2float(a0) * __half2float(b0)
                         + __half2float(a1) * __half2float(b1);
                }
            }
        }

        __syncthreads();  // Before loading next tile
    }

    if (m < M) {
        C[M * l + m] = __float2half(acc);
    }
}

void nvfp4_gemv(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {
    const int64_t M = a.size(0);
    const int64_t L = a.size(2);
    const int64_t K_pairs = a.size(1);
    const int64_t K_blocks = sfa.size(1);

    const uint8_t* a_ptr = static_cast<const uint8_t*>(a.data_ptr());
    const uint8_t* b_ptr = static_cast<const uint8_t*>(b.data_ptr());
    const uint8_t* sfa_ptr = static_cast<const uint8_t*>(sfa.data_ptr());
    const uint8_t* sfb_ptr = static_cast<const uint8_t*>(sfb.data_ptr());
    auto* c_ptr = static_cast<at::Half*>(c.data_ptr());

    int threadsPerBlock = 32;
    dim3 numBlocks(CEIL_DIV(M, threadsPerBlock), 1, L);

    gemv<<<numBlocks, threadsPerBlock>>>(
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        static_cast<int>(M),
        static_cast<int>(K_pairs),
        static_cast<int>(K_blocks),
        static_cast<int>(L),
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(1), b.stride(2),
        sfa.stride(0), sfa.stride(1), sfa.stride(2),
        sfb.stride(1), sfb.stride(2)
    );
}

"""

_ext = load_inline(
    name="nvfp4_gemv_ext",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["nvfp4_gemv"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemv
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float4e2m1fn] of shape [m, k, l],
            b: torch.Tensor[float4e2m1fn] of shape [1, k, l],
            sfa: torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l], used by reference implementation
            sfb: torch.Tensor[float8_e4m3fnuz] of shape [1, k // 16, l], used by reference implementation
            sfa_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_m, 4, rest_k, l],
            sfb_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_n, 4, rest_k, l],
            c: torch.Tensor[float16] of shape [m, 1, l]
    Returns:
        Tensor containing output in float16
        c: torch.Tensor[float16] of shape [m, 1, l]
    """
    # c: [l, m, 1] is pre-allocated memory to avoid timing allocation overhead.
    a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data

    # Your implementation here

    # Call compiled launcher (it gets tensors, not raw pointers):
    _ext.nvfp4_gemv(a, b, sfa, sfb, c)

    return c
