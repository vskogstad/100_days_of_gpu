from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>

// __global__ device kernel(s) here

// C++ host launcher that takes torch::Tensor, gets raw pointers, launches kernels
void nvfp4_gemv_launcher(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {
    // get sizes, set up grid/block, call <<<>>> on __global__ kernel
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemv", &nvfp4_gemv_launcher, "NVFP4 GEMV kernel");
}
"""

_ext = load_inline(
    name="nvfp4_gemv_ext",
    cpp_sources="",
    cuda_sources=[source],   # <-- IMPORTANT: use cuda_sources
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
