Day 1: Read first chapter of "Programming Massively Parallel Processors". Simple vector addition in cuda on leetgpu, then repeated on colab with memory management.

Day 2: Working through elementwise activations. ReLU, leaky ReLU, GELU and HardTanh. 

Day 3: Working on sum/Reduction. Got a working LeetGPU implementation for now, have to continue later. Doing final addition from each block on the CPU.

Day 4: Continued on Sum/reduction. Adding tail shuffle and grid-strided loading. SiLU kernel as review. Read chapter 2 of PMPP.

Day 5: Read Chapter 3 in PMPP. Made a single block implementation of prefix-Sum. Will have to continue tomorrow. 

Day 6: Halfway through chapter 4 in PMPP. Completed cross-block implementaion of prefix-Sum.

Day 7: Completed chapter 4. Stuck for hours on the hierarchical approach for generalizing prefix-Sum. Today felt like attempting to cram to many techniques/concepts with a shaky base understanding. Implemented a basic matrix add kernel.

Day 8: Read the first quarter of https://www.aleksagordic.com/blog/matmul. Implemented a tiled matrix multiply. New concepts: Templating, Memory coalescence, use of __restrict__ and #pragma unroll

Day 9: Tiled matrix transpose. Happy with results today, cleared up some of my misunderstandings with coalescing and managed to mostly reason about how to solve this problem efficiently on my own. New concepts: Bank conflicts.

Day 10: Naive matrix copy. Did not manage to get loading in float4 to minimize overhead. A bit of a let down after good progress yesterday. Will continue tomorrow.

Day 11: Tiled vectorized matrix copy. Vectorized RGB-invert on leetcode. Watched 3blue1brown video on convolution. Implementing tomorrow.

Day 12: Naive convolution and in-place array reversal.

Day 13: Dynamic shared memory and usage of __constant__ for 1D-convolution.

Day 14: Back to array reversal. Tiled register swap instead of naive baseline.

Day 15: Naive SwiGLU kernel. Short on time today.

Day 16: Micro optimizations on SwiGLU kernel. Single block naive RMSNorm kernel.