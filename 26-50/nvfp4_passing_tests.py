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

__global__ void gemv(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ sfa,
    const uint8_t* __restrict__ sfb,
    at::Half* __restrict__ C,
    int M, int K, int L,
    int a_s0,
    int a_s1,
    int a_s2,

    int b_s0,
    int b_s1,
    int b_s2,

    int sfa_s0,
    int sfa_s1,
    int sfa_s2,

    int sfb_s0,
    int sfb_s1,
    int sfb_s2,
    
    int K_pairs,
    int K_blocks,
    int pairs_per_block
) {
    // treat bytes as FP4x2 storage
    auto* a_fp4x2 = reinterpret_cast<const __nv_fp4x2_storage_t*>(A);
    auto* b_fp4x2 = reinterpret_cast<const __nv_fp4x2_storage_t*>(B);

    // then use __nv_cvt_fp4x2_to_halfraw2(...) to get half2

    int tid = threadIdx.x;
    int m   = blockIdx.x * blockDim.x + tid;
    int l   = blockIdx.z;   // grid.z indexes L

    if (m >= M || l >= L) return;

    int numel = M * L;

    int C_idx = M*l + m;
        
    float acc = 0.0f;
    for (int pair = 0; pair < K_pairs; pair++) {
        // order of memory [m, l, k]
        int block_k = pair/pairs_per_block;
        // pair is your “k along fp4x2” index

        // A: [M, K_pairs, L]
        int64_t A_idx = m * a_s0 + pair * a_s1 + l * a_s2;

        // B: [1, K_pairs, L]
        int64_t B_idx =         pair * b_s1 + l * b_s2;

        // sfa: [M, K_blocks, L]
        int64_t sfa_idx = m * sfa_s0 + block_k * sfa_s1 + l * sfa_s2;

        // sfb: [1, K_blocks, L]
        int64_t sfb_idx =              block_k * sfb_s1 + l * sfb_s2;
        //int A_idx   = m*L*K_pairs + l*K_pairs + pair;  
        //int sfa_idx = m*L*K_blocks + l*K_blocks + block_k;
        //int B_idx   = l*K_pairs + pair;
        //int sfb_idx = l*K_blocks + block_k;
        
        // fp4/fp8 conversion:
        __nv_fp4x2_storage_t a_pair = a_fp4x2[A_idx];
        __nv_fp4x2_storage_t b_pair = b_fp4x2[B_idx];

        // 2) Decode FP4x2 -> half2 (two fp16 values)
        __half2_raw h2_raw = __nv_cvt_fp4x2_to_halfraw2(a_pair, __NV_E2M1);
        __half2 h2 = *reinterpret_cast<__half2*>(&h2_raw);

        __half a0 = __low2half(h2);
        __half a1 = __high2half(h2);
        
        
        __half2_raw h2b_raw = __nv_cvt_fp4x2_to_halfraw2(b_pair, __NV_E2M1);
        __half2 h2b = *reinterpret_cast<__half2*>(&h2b_raw);

        __half b0 = __low2half(h2b);
        __half b1 = __high2half(h2b);
        
        

        const __nv_fp8_storage_t* Sa_fp8 =
            reinterpret_cast<const __nv_fp8_storage_t*>(sfa);
        __nv_fp8_storage_t sa_code = Sa_fp8[sfa_idx];
        __half_raw sa_raw = __nv_cvt_fp8_to_halfraw(sa_code, __NV_E4M3);
        __half sa = *reinterpret_cast<__half*>(&sa_raw);
        
        const __nv_fp8_storage_t* Sb_fp8 =
            reinterpret_cast<const __nv_fp8_storage_t*>(sfb);
        __nv_fp8_storage_t sb_code = Sb_fp8[sfb_idx];
        __half_raw sb_raw = __nv_cvt_fp8_to_halfraw(sb_code, __NV_E4M3);
        __half sb = *reinterpret_cast<__half*>(&sb_raw);

        // 4) Apply scale(s). If there were a tensor fp32 scale S_tensor, you’d fold it in here.
        a0 = __hmul(a0, sa);
        a1 = __hmul(a1, sa);
        
        b0 = __hmul(b0, sb);
        b1 = __hmul(b1, sb);
    
    
        acc += __half2float(a0) * __half2float(b0)
             + __half2float(a1) * __half2float(b1);

    }

    if (C_idx < numel) {
        C[C_idx] = __float2half(acc);
    }
}


// C++ host launcher that takes torch::Tensor, gets raw pointers, launches kernels
void nvfp4_gemv(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {
    // get sizes, set up grid/block, call <<<>>> on __global__ kernel
    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t L = a.size(2);

    // Use untyped data_ptr() (no template) to avoid shell dtype ABI issues
    const uint8_t* a_ptr   = static_cast<const uint8_t*>(a.data_ptr());
    const uint8_t* b_ptr   = static_cast<const uint8_t*>(b.data_ptr());
    const uint8_t* sfa_ptr = static_cast<const uint8_t*>(sfa.data_ptr());
    const uint8_t* sfb_ptr = static_cast<const uint8_t*>(sfb.data_ptr());
    auto*          c_ptr   = static_cast<at::Half*>(c.data_ptr());

    int threadsPerBlock = 512;
    int T = threadsPerBlock;
    
    const int64_t a_s0 = a.stride(0);
    const int64_t a_s1 = a.stride(1);
    const int64_t a_s2 = a.stride(2);

    const int64_t b_s0 = b.stride(0);  // dim0 = 1
    const int64_t b_s1 = b.stride(1);
    const int64_t b_s2 = b.stride(2);

    const int64_t sfa_s0 = sfa.stride(0);
    const int64_t sfa_s1 = sfa.stride(1);
    const int64_t sfa_s2 = sfa.stride(2);

    const int64_t sfb_s0 = sfb.stride(0);  // dim0 = 1
    const int64_t sfb_s1 = sfb.stride(1);
    const int64_t sfb_s2 = sfb.stride(2);
    
    
    const int64_t K_pairs  = a.size(1);      // number of fp4x2 entries along K
    const int64_t K_blocks = sfa.size(1);    // number of FP8 scales along K

    const int64_t pairs_per_block = K_pairs / K_blocks;

    // 3D grid: x over M, y unused for now, z over L
    dim3 numBlocks(CEIL_DIV(M, T), 1, L);

    gemv<<<numBlocks, threadsPerBlock>>>(
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(L),
        a_s0,
        a_s1,
        a_s2,

        b_s0,
        b_s1,
        b_s2,

        sfa_s0,
        sfa_s1,
        sfa_s2,

        sfb_s0,
        sfb_s1,
        sfb_s2,
        
        K_pairs,
        K_blocks,
        pairs_per_block
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
