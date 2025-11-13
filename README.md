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

Day 17: Unfinnished improvements on RMSNorm kernel. Reusing binary reduction and warp shuffling from reduction kernel.

Day 18: Improved RMSNorm kernel to 98th percentile on LeetGPU. Very happy with that. 

Day 19: Count array element. Buggy, fails for large N.

Day 20: Got count array element to pass somehow. 

Day 21: Improved count array element. Should have just used atomicAdd from the start. Naive single block softmax. 

Day 22: Ironed out some bugs in my softmax-kernel. Added warp shuffling. Attempted to cache, but see no speedups.

Day 23: Online softmax, single block.

Day 24: Better guards, incomplete warp-shuffle.

Day 25: Warp shuffle fixed. Max warp occupancy, 57th percentile. Moving on to softmax attention.

Day 26: Not much progress on attention kernel today. Tried to understand how to properly tile the problem in a way that would be efficient and understandable both for the matmul part and the softmax. I think I might be subconciously stalling by watching/reading passively instead of trying and failing to implement. 

Day 27: Continued with attention kernel. Ended up with grouping each row of Q into separate block. Code is non-functional but basic structure is there. 

Day 28: Working attention kernel! Slow as I am using a lot of atomicAdd directly to the output-matrix. Next I need to accumulate in shared memory before writing to output.

Day 29: Tiled write to shared memory then atomicAddto output matrix. Some improvement but not much. The best attention implementations on leetGPU are still 20x faster. Need to go through those step by step and try to understand how they work.

Day 30: Not very bright today. Attempt at dot-product kernel instead of continuing on attention, but still struggled beyond naive implementation.

Day 31: Matrix add.