import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input + offsets, mask=mask, other=-float("inf"))

    x_row = x - tl.max(x)
    o = tl.exp(x_row) / tl.sum(tl.exp(x_row))

    tl.store(output + offsets, o, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = triton.next_power_of_2(N)
    n_blocks = 1  # single block, small problems only
    grid = (n_blocks,)
    softmax_kernel[grid](input, output, N, BLOCK_SIZE)
