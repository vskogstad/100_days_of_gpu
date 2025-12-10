import torch
import triton
import triton.language as tl

@triton.jit
def matrix_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE) 

    mask = offsets < n_elements

    a_b = tl.load(a + offsets, mask=mask)
    b_b = tl.load(b + offsets, mask=mask)
    c_b = tl.load(c + offsets, mask=mask)

    c_b = a_b + b_b

    tl.store(c + offsets, c_b, mask=mask)
   
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
