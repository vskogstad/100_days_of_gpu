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

Day 32: GEMM. Not finished.

Day 33: Did a study session on siboehms mamtmul block, worked a little bit more on GEMM kernel.

Day 34: Completed two first parts of GEMM kernel. Working on shared memory.

Day 35: Shared memory working. Roughly same speed on LeetGPU.

Day 36: Read through kernel 4 on [the blogpost](https://siboehm.com/articles/22/CUDA-MMM)/chatted with GPT about 1D-blocktiling but didn't actually implement. Need to better balance both trying to actually understand what I'm doing, but not use lack of clear understanding as a crutch to delay writing code. I think I sometimes do the latter, though you typically get at least some more understanding from trying to implent. 
Implemented the (easy) rainbow table kernel on leetGPU.

Day 37: Implemented columnwise 1D blocktiling (kernel 4). Also loading A_s transposed.

Day 38: Implemented 2D blocktiling to GEMM. Implemented an int8 matmul with scaling factors and zero-point adjustment.

Day 39: Implemented Batched matrix multiply on leetGPU today. Batch dimension ended up not really being much of a factor, so it was mainly repetition from earlier days. Probably neccessary, as I am still not very confident about indexing after threadwise tiling. 

Day 40: Not written a lot of code today, but did an hour of kernel study group, then chatting with LLM to set up a template for running CUDA code from pytorch/CuTe templates provided for the NVFP4 challenge. Almost all of the code for today is LLM-written, so maybe this is my first day with no kernels? Learned about load_inline.

Day 41: More attempts at getting inline CUDA to work. Not compiling yet.

Day 42: Some more work on getting inline CUDA up and running. Did a FP16 Batched Matmul leetGPU kernel.

Day 43: First NVFP4 kernel passing all tests! Code is a mess of me and 5.1 iterating back and forth but feels really good to have made progress.

Day 44: Was able to get cuda NVFP4 down to just about exactly the same time as the pytorch baseline ~150 us. Heavy LLM usage. With just one more day left to submit, I think I might have lost sight of what I should be targeting from the challenge (learning). As iterations pile on and the codebase gets more involved, it's too tempting to be playing LLM slots, hoping claude will come up with a better solution on the next prompt. On a postive note, I've still learned a lot, but I would definitely not be able to rewrite the kernel again from scratch.
Gonna try to watch the GPU-mode video on CuTe and try to submit a solution using that tomorrow.

Day 45: Spent today mostly as planned. Going through the video and then the reference kernel line by line with ChatGPT. Took more time than anticipated, but feelt worthwile. Haven't written any new code today. Going to go through the blog posts covering the kernel and improvements tomorrow.

Day 46: Experimenting a little bit with the launch grid patterns. Get some good speedups just by decreasing tpb. 

Day 47:
So I didn't produce any code today. Watched Tri Daos video on CuteDSL. Looked at the nvfp4 - atomic add example which parallelizes along the k-dimension but did not implement it myself. It's not conceptually different from what I've done before, but the CuTe-syntax is still foreign so I have to basically copy paste line by line when trying to reimplement... Started looking at tensor cores but did not understand much.

Day 48:
Today I watched the GPU_mode video on tensor cores and started going through the CuTe blackwell fp16 tutorial gemm line by line. Both pipelining and tensor cores are new, so need some time to work through. Feels like I'm making some progress, but will have to implement and repeat it again later.

Day 49:
Back from a week of vacation. Not much kernel-progress today, but did a lot of work in pytorch. Watched through a triton-video and the first half of the latest GPU MODE video regarding the NVFP4 contest. Honestly found it a bit hard to follow. 

Day 50:
Attempted softmax kernel in triton. 

Day 51:
Just working through basic triton kernels. Vector add, matrix add, reverse array.

Day 52: 
Naive softmax kernel in triton, then watched relevant part of Umar Jamil video on flash attention afterwards.

Day 53:
Attention as an autograd function in pytorch. Implemented the forward pass.

Day 54:
Flash-attention as an autograd function in pytorch. Forward only. Triton setup.

Day 55: 
Translated my pytorch flash-attention kernel into triton. Not passing tests yet.

Day 56: 
Flash-Attention forward with causal mask in Triton. Super happy with that. Backward pass in pytorch next. 

Day 57: 
Set up the basic structure of the backward pass with recomputation in pytorch. Spent probably 2 hours on deriving the backward pass of flash attention on paper. Had a lot of help from Claude. 

Day 58:
Working flash attention backward pass in pytorch. Started setting up basic benchmarking code to compare against torch.nn.functional and pytorch with torch.compile().

Day 59:
Implemented non-tiled backward pass in pytorch. Benchmarking of forward, backward and both. Testing on leaderboard. 27.4ms.