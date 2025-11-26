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

__global__ void gemv_warp_kb_vec(
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
    extern __shared__ uint8_t smem[];
    __nv_fp4x2_storage_t* B_sh = reinterpret_cast<__nv_fp4x2_storage_t*>(smem);
    __nv_fp8_storage_t* sfb_sh = reinterpret_cast<__nv_fp8_storage_t*>(smem + K_pairs);

    auto* a_fp4x2 = reinterpret_cast<const __nv_fp4x2_storage_t*>(A);
    auto* b_fp4x2 = reinterpret_cast<const __nv_fp4x2_storage_t*>(B);
    auto* Sa_fp8 = reinterpret_cast<const __nv_fp8_storage_t*>(sfa);
    auto* Sb_fp8 = reinterpret_cast<const __nv_fp8_storage_t*>(sfb);

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int m = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int l = blockIdx.z;

    // Cooperative load B and sfb
    for (int i = tid; i < K_pairs; i += blockDim.x) {
        int64_t B_idx = i * b_s1 + l * b_s2;
        B_sh[i] = b_fp4x2[B_idx];
    }
    for (int i = tid; i < K_blocks; i += blockDim.x) {
        int64_t sfb_idx = i * sfb_s1 + l * sfb_s2;
        sfb_sh[i] = Sb_fp8[sfb_idx];
    }
    __syncthreads();

    if (m >= M) return;

    float acc = 0.0f;

    // Check if A is contiguous along K dimension (a_s1 == 1)
    const bool a_contig = (a_s1 == 1);

    for (int kb = lane_id; kb < K_blocks; kb += WARP_SIZE) {
        // Load scale factors once per K_block
        int64_t sfa_idx = m * sfa_s0 + kb * sfa_s1 + l * sfa_s2;
        __half_raw sa_raw = __nv_cvt_fp8_to_halfraw(Sa_fp8[sfa_idx], __NV_E4M3);
        float sa = __half2float(*reinterpret_cast<__half*>(&sa_raw));
        
        __half_raw sb_raw = __nv_cvt_fp8_to_halfraw(sfb_sh[kb], __NV_E4M3);
        float sb = __half2float(*reinterpret_cast<__half*>(&sb_raw));
        
        float scale = sa * sb;

        int pair_base = kb * PAIRS_PER_SCALE;

        if (a_contig) {
            // Vectorized path: load all 8 A values at once (8 bytes = uint64_t)
            int64_t A_base = m * a_s0 + pair_base + l * a_s2;
            uint64_t a_packed = *reinterpret_cast<const uint64_t*>(A + A_base);
            
            // Also load 8 B values from shared
            uint64_t b_packed = *reinterpret_cast<const uint64_t*>(&B_sh[pair_base]);

            #pragma unroll
            for (int t = 0; t < PAIRS_PER_SCALE; t++) {
                __nv_fp4x2_storage_t a_pair = (a_packed >> (t * 8)) & 0xFF;
                __nv_fp4x2_storage_t b_pair = (b_packed >> (t * 8)) & 0xFF;

                __half2_raw h2a_raw = __nv_cvt_fp4x2_to_halfraw2(a_pair, __NV_E2M1);
                __half2 h2a = *reinterpret_cast<__half2*>(&h2a_raw);
                
                __half2_raw h2b_raw = __nv_cvt_fp4x2_to_halfraw2(b_pair, __NV_E2M1);
                __half2 h2b = *reinterpret_cast<__half2*>(&h2b_raw);

                float a0 = __half2float(__low2half(h2a));
                float a1 = __half2float(__high2half(h2a));
                float b0 = __half2float(__low2half(h2b));
                float b1 = __half2float(__high2half(h2b));

                acc += scale * (a0 * b0 + a1 * b1);
            }
        } else {
            // Scalar fallback
            #pragma unroll
            for (int t = 0; t < PAIRS_PER_SCALE; t++) {
                int pair = pair_base + t;
                
                int64_t A_idx = m * a_s0 + pair * a_s1 + l * a_s2;
                __nv_fp4x2_storage_t a_pair = a_fp4x2[A_idx];
                __nv_fp4x2_storage_t b_pair = B_sh[pair];

                __half2_raw h2a_raw = __nv_cvt_fp4x2_to_halfraw2(a_pair, __NV_E2M1);
                __half2 h2a = *reinterpret_cast<__half2*>(&h2a_raw);
                
                __half2_raw h2b_raw = __nv_cvt_fp4x2_to_halfraw2(b_pair, __NV_E2M1);
                __half2 h2b = *reinterpret_cast<__half2*>(&h2b_raw);

                float a0 = __half2float(__low2half(h2a));
                float a1 = __half2float(__high2half(h2a));
                float b0 = __half2float(__low2half(h2b));
                float b1 = __half2float(__high2half(h2b));

                acc += scale * (a0 * b0 + a1 * b1);
            }
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

    gemv_warp_kb_vec<<<grid, threadsPerBlock, smem_size>>>(
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
