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

cuda_src = r"""#include <torch/extension.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define CEIL_DIV(A, B) ((A + B - 1) / B)
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define PAIRS_PER_SCALE 8

__global__ void gemv_fast(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ sfa,
    const uint8_t* __restrict__ sfb,
    at::Half* __restrict__ C,
    int M, int K_pairs, int K_blocks, int L,
    int a_s0, int a_s2,
    int b_s2,
    int sfa_s0, int sfa_s2,
    int sfb_s2
) {
    extern __shared__ uint8_t smem[];
    uint8_t* B_sh = smem;
    uint8_t* sfb_sh = smem + K_pairs;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int m = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int l = blockIdx.z;

    // Precompute base pointers for this (m, l) - ONCE per thread
    // A layout: [M, K_pairs, L] with stride [a_s0, 1, a_s2]
    const uint8_t* A_row = A + m * a_s0 + l * a_s2;
    const uint8_t* sfa_row = sfa + m * sfa_s0 + l * sfa_s2;
    
    // B layout: [1, K_pairs, L]
    const uint8_t* B_base = B + l * b_s2;
    const uint8_t* sfb_base = sfb + l * sfb_s2;

    // Cooperative load B and sfb - use simple linear indexing
    for (int i = tid; i < K_pairs; i += blockDim.x) {
        B_sh[i] = B_base[i];
    }
    for (int i = tid; i < K_blocks; i += blockDim.x) {
        sfb_sh[i] = sfb_base[i];
    }
    __syncthreads();

    if (m >= M) return;

    float acc = 0.0f;

    // Main loop - minimal index calculations
    for (int kb = lane_id; kb < K_blocks; kb += WARP_SIZE) {
        // Scale factors - simple byte offset
        float sa = __half2float(*reinterpret_cast<const __half*>(
            &__nv_cvt_fp8_to_halfraw(sfa_row[kb], __NV_E4M3)));
        float sb = __half2float(*reinterpret_cast<const __half*>(
            &__nv_cvt_fp8_to_halfraw(sfb_sh[kb], __NV_E4M3)));
        float scale = sa * sb;

        // Base offset for this K_block's pairs
        int pair_base = kb * PAIRS_PER_SCALE;  // kb << 3
        
        // Load 8 bytes of A at once
        uint64_t a_packed = *reinterpret_cast<const uint64_t*>(A_row + pair_base);
        uint64_t b_packed = *reinterpret_cast<const uint64_t*>(B_sh + pair_base);

        // Process all 8 pairs
        #pragma unroll
        for (int t = 0; t < PAIRS_PER_SCALE; t++) {
            uint8_t a_byte = (a_packed >> (t * 8)) & 0xFF;
            uint8_t b_byte = (b_packed >> (t * 8)) & 0xFF;

            __half2_raw h2a_raw = __nv_cvt_fp4x2_to_halfraw2(a_byte, __NV_E2M1);
            __half2_raw h2b_raw = __nv_cvt_fp4x2_to_halfraw2(b_byte, __NV_E2M1);
            
            float a0 = __half2float(*reinterpret_cast<__half*>(&h2a_raw.x));
            float a1 = __half2float(*reinterpret_cast<__half*>(&h2a_raw.y));
            float b0 = __half2float(*reinterpret_cast<__half*>(&h2b_raw.x));
            float b1 = __half2float(*reinterpret_cast<__half*>(&h2b_raw.y));

            acc += scale * (a0 * b0 + a1 * b1);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
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

    int threadsPerBlock = WARPS_PER_BLOCK * WARP_SIZE;
    dim3 grid(CEIL_DIV(M, WARPS_PER_BLOCK), 1, L);
    
    size_t smem_size = K_pairs + K_blocks;

    gemv_fast<<<grid, threadsPerBlock, smem_size>>>(
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        static_cast<int>(M),
        static_cast<int>(K_pairs),
        static_cast<int>(K_blocks),
        static_cast<int>(L),
        static_cast<int>(a.stride(0)),
        static_cast<int>(a.stride(2)),
        static_cast<int>(b.stride(2)),
        static_cast<int>(sfa.stride(0)),
        static_cast<int>(sfa.stride(2)),
        static_cast<int>(sfb.stride(2))
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
