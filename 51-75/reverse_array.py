import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(
    input,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    rev_offsets = N - 1 - offsets

    mask = offsets < (N // 2)


    left_part = tl.load(input + offsets, mask=mask)
    right_part = tl.load(input + rev_offsets, mask=mask)

    tl.store(input+offsets, right_part, mask=mask)
    tl.store(input+rev_offsets, left_part, mask=mask)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)
    
    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    ) 