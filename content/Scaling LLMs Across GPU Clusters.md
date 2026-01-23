Model Training can be derived into 3 steps:
1. Forward Pass - inputs through model to yield outputs
2. Backward Pass - compute gradients
3. Optimization Step - update parameters via gradient descent

![[Screenshot 2026-01-19 at 9.51.51 AM.png]]

The *batch size* (bs) is one of the important *hyperparameters* for model training; it affects both model convergence and throughput.

Small batch size -> noisy gradients, model may not converge to optimal final performance
- requires more optimizer steps, optimizer steps are expensive in compute + add total time to train

Large batch size -> less importance per training token, slower convergence
- more accurate gradient estimations, potentially waste compute resources

Traditionally, batch size can be tweaked a lot around the optimal batch size, without largely affecting model quality.

In LLM pretraining, batch sizes are usually represented in terms of tokens, instead of number of samples. Training on a single machine, the *bs* (in samples) and *bst* can be computed from the model input sequence length (*seq*) as:
$$
bst = bs \cdot t
$$
The sweet spot for LLM training is 4-60 million tokens.

`llama1: bs ~ 4 million tokens; 1.4 trillion total tokens`

`deepseekR1: bs ~ 60 million tokens; 14 trillion total tokens`


The first problem we run into **Out Of Memory(OOM)** issues. What should we do when our GPU doesn’t have enough memory to hold a full batch of our target batch size?

When training a neural network you store:
- Model Weights
- Model Gradients
- Optimizer States
- Activations needed to compute the gradients

All of those are stored in *tensors* which come in different shapes and precisions 
- The _shapes_ are determined by hyperparameters such as batch size, sequence length, model hidden dimensions, attention heads, vocabulary size, and potential model sharding
- *Precision* refers to formats like FP32, BF16, or FP8, which respectively require 4, 2, or 1 byte to store each single value in the tensor.

For a simple transformer based LLM, the number of parameters is given as:
$$
N = h \cdot v + L(1/2h^2+1/3h^2)+2h
$$
where: 

N = Total Parameters,
h = Hidden Dimensions,
v = Vocab Size,
L = Number of Layers,

Here's how that equation breaks down:

**1. Token Embedding Matrix:**
$$
h\cdot v
$$
- Shape: `[Vocab Size, Hidden Dimensions]`
- Each token has an embedding vector of length `h`
- Total Parameters: Vocab Size x Hidden Dimensions

**2. Transformer Layers:**
$$
L(1/2h^2+1/3h^2)
$$
This term counts *per-layer parameters*, then multiplies by `L`. (How many parameters per layer x Total Number of Layers)
- Attention Projections:
  - From multi-head self-attention:
    - Q, K, V projections
    - Output projection
  - Each is roughly `h × h`, but sharing / packing reduces constants → ~½ h²
- Feed Forward network 
  - Typical FFN:
    - First linear: `h → 4h`
    - Second linear: `4h → h`
  - Total ≈ `8h²`, but with architectural simplifications and constant folding, it's approximated as `⅓ h²`.

**3. LayerNorm Parameters**
$$
2h
$$
Per layer:
- LayerNorm scale (γ): `h`
- LayerNorm bias (β): `h`

Memory requirements for the parameters and gradients are determined by multiplying the number of parameters by the number of bytes per parameter. In full precision (FP32) training, both parameters and gradients require 4 bytes while the optimizer, if we use Adam, requires the momentum and variance to be stored, adding another 8 bytes per parameter

Mathematically, that is:
$$
\begin{aligned}
m_{\text{params}} &= 4N \\
m_{\text{grad}} &= 4N \\
m_{\text{opt}} &= (4 + 4)N
\end{aligned}
$$

This is a naive full low precision approach.

Instead, we use a mixed precision approach which:
- Use BF16 for parameters and gradients, which would be 2 bytes per parameter/gradient
- An additional copy of parameters and gradients stored in FP32, which would be 4 bytes per parameter/gradient (also known as master weights)
- Optimizer states via Adam storing momentum + variance in FP32, each being 4 bytes.

Mathematically that is:
$$
\begin{aligned}
m_{\text{params}} &= 2 \cdot N \\
m_{\text{grad}} &= 2 \cdot N \\
m_{\text{params\_fp32}} &= 4 \cdot N \\
m_{\text{opt}} &= (4 + 4) \cdot N
\end{aligned}
$$
Mixed precision doesn't directly reduce memory, it adds 4 bytes of data over full precision. It instead provides three massive advantages:
1. Compute the forward/backward passes in half precision
2. Allows us to use optimized lower precision operations on the GPU, which are faster
3. Reduces the activation memory requirements during the forward pass

Now that we know how we store our parameters, let's move onto activation memory.

Activation memory can be calculated a little differently than how we calculated memory for storing our parameters. The formula follows:

$$
m_{\text{act}} = L \cdot \text{seq} \cdot \text{bs} \cdot h \cdot \left(3/4 + \frac{5 \cdot n_{\text{heads}} \cdot \text{seq}}{h}\right)
$$

Where: 

*L* = number of layers, 
*seq* = sequence length, 
*bs* = batch size, 
*h* = hidden dimensions, 
$n_{\text{heads}}$ = number of heads

Memory usage is not static for a given model; rather, it scales linearly with the batch size and quadratically with the sequence length. Activation memory is the part that will blow up when we increase our batch size or train with longer sequences.
![[Screenshot 2026-01-19 at 2.54.09 PM.png]]

For short sequences (or small batch sizes), memory usage for activations is almost negligible, but from around 2-4k tokens they start to take up a significant amount of memory, while usage for parameters, gradients, and optimizer states is roughly independent of the sequence length and batch size.

# Activation Recomputation
Activation Recomputation is also known as gradient checkpointing. The general idea behind activation recomputation is to discard some activations during the forward pass to save memory and spend some extra compute to recompute these on the fly during the backward pass.
![[Screenshot 2026-01-19 at 3.08.38 PM.png]]
Instead of storing every single activation, we discard some of the activations, and then recompute on the fly. This adds computation, but saves memory.

There are a few strategies for selecting key activations to store:

- **Full**: Checkpoint activations at the transition point between each layer of the Transformer model. It requires a forward pass through each layer, essentially adding a full forward pass during the backward pass. Saves the most memory but is the most expensive one in terms of compute. It typically increases the compute cost and time by up to 30-40%


- **Selective**: Discard the attention computations. Focus on checkpointing the expensive feedforward computations. **70% activation memory reduction at a 2.7% compute cost**.

**Activation recomputation slightly increases the number of FLOPS due to recomputation, while it significantly reduces memory access overhead.**

This trade-off is particularly advantageous on hardware with limited high-speed memory, like GPUs, as accessing memory is typically slower than performing computations. Despite the additional operations involved, the overall effect is thus often faster computation, in addition to the much lower memory footprint.


# Gradient Accumulation
Gradient accumulation is a very straightforward method that consists of splitting a batch into micro-batches. Perform forward and backward passes successively on each micro-batch, compute the gradients, and then sum the gradients of all micro-batches before we perform optimization.
![[Screenshot 2026-01-19 at 3.41.54 PM.png]]
Gradient accumulation allows us to reduce activation memory by processing smaller micro-batches sequentially. This reduces stored activations and gradients since only one micro-batch's worth of activations needs to be kept in memory at a time.

However, gradient accumulation requires multiple consecutive forward/backward passes per optimization step, thereby increasing the compute overhead and slowing down training. Another approach is to perform the forward pass and backward pass of each microbatch in parallel without waiting for each batch. This leads us to ...

# Data Parallelism
Data parallelism (DP) simply put replicates the model on several GPUs (we call the replicas “model instances”) and run forward and backward passes on different micro-batches of data in parallel on each GPU.
![[Pasted image 20260119155418.png]]

Different micro-batch for each GPU means we’ll have different gradients on each GPU, so to keep the model instances in sync across the different GPUs, we average the gradients from the model instances via “all-reduce.” This operation takes place during the backward pass, before the optimizer step.

![[Screenshot 2026-01-19 at 3.59.40 PM.png]]

The naive implementation of "all-reduce" is to wait for all backward passes to finish, so that we have all the gradients, then trigger an all-reduce over all the DP ranks to sync the gradients. This is horrible and leads to GPU's just sitting around doing nothing! **SUPER BAD**.

A better way to do this operation is to sync them as each gradient is computed. For example, as soon as the backward pass of the last layer is complete, those gradients can already be gathered and summed while the backward computations continue for earlier layers, moving toward the left. Here's what that looks like:
![[Screenshot 2026-01-19 at 4.23.20 PM.png]]

Our next step derives from the fact that GPU operations are usually more efficient when performed on large tensors, rather than having many operations running on smaller tensors. Group gradients into “buckets” and launch a single all-reduce for all the gradients within the same bucket instead of performing independent all-reduce operations for each gradient. Here's what that looks like:
![[Screenshot 2026-01-19 at 4.45.59 PM.png]]
Think of it like packing items into boxes before shipping them. It's more efficient to send a few big boxes than many small ones. By performing a single all-reduce operation for each bucket, we can significantly reduce the communication overhead and speed up the communication operation.


Here's everything we've gone through so far:

1. We first determine the best (global) batch size in tokens, either by consulting the literature or by running experiments measuring model convergence.
2. We then select a sequence length for training, again by either consulting the literature or running experiments. Generally, 2-8k tokens works reliably well for the evaluation benchmarks we have today.
3. We now know the batch size (*gbs*). We can find the maximum local batch size (*mbs*) on a single GPU by increasing the local batch size until we run out of memory.
4. Finally, we determine the number of available GPUs for our target *dp*. The ratio of *gbs* to *dp* gives us the remaining number of gradient accumulation steps needed for the desired *gbs*.

If the gradient accumulation ratio is lower than 1 - i.e., we have too many GPUs/are GPU-rich - we can either choose not to use all our GPUs, explore a larger *gbs*, or test if a lower *mbs* will speed up training. In the latter case we’ll end up prioritizing throughput over individual GPU compute efficiency, using a smaller *mbs* than possible in order to speed up training.

Data parallelism starts to have some limiting communication overhead above a certain level of scaling.

There are two main approaches to splitting: parallelism (tensor, context, or pipeline parallelism) and sharding (DeepSpeed ZeRO or PyTorch FSDP).


### Zero Redundancy Optimizer (ZeRO)

While data parallelism is an efficient way to scale training, the naive replication of optimizer states, gradients, and parameters across each DP rank introduces significant memory redundancy. ZeRO eliminates this by partitioning the optimizer states, gradients, and parameters across the data parallel dimension, while still allowing computation with the full set of parameters.

This approach is organized into three possible optimization stages:
- ZeRO-1: optimizer state partitioning
- ZeRO-2: optimizer state + gradient partitioning
- ZeRO-3: optimizer state + gradient + parameter partitioning


Given model's parameter count `Ψ` (previously `N`):
- Model’s parameters (half precision; i.e., BF16/FP16): 2Ψ
- Model’s gradients (half precision; i.e., BF16/FP16): 2Ψ
- Model’s parameters in FP32 and optimizer states: 4Ψ+(4Ψ+4Ψ)
- Model’s gradients in FP32: 4Ψ (optional, only included if we want to accumulate gradients in FP32)

The idea of ZeRO is to shard these objects across the DP ranks, with each node only storing a slice of the items. These slices are then reconstructed when and if needed, thereby dividing memory usage by the data parallel degree.

![[Screenshot 2026-01-20 at 4.51.21 PM.png]]

In vanilla DP, all ranks gather the same gradients after the backward pass and simultaneously perform identical optimizer steps. In ZeRO-1, optimizer states are sharded across data-parallel ranks. Given a data-parallel degree $N_d$, each rank stores and updates only a $\frac{1}{N_d}$ fraction of the optimizer states. Consequently, during the optimization step, only $\frac{1}{N_d}$ of the FP32 parameters are updated per rank.

However, during the forward pass, each data-parallel replica must hold the complete model parameters. Since only a $\frac{1}{N_d}$ shard of the FP32 weights is updated on each rank during optimization, an additional *all-gather* operation is required after the optimizer step to assemble the full set of updated parameters on every replica.

A single training step proceeds as follows. First, each data-parallel replica performs a forward pass using the same full set of BF16 model parameters, but on different micro-batches of data. Next, each replica executes a backward pass, producing gradients corresponding to its local micro-batch. These gradients are then aggregated across replicas using a *reduce-scatter* collective operation. *Reduce-scatter* is a collective communication operation that aggregates tensors across all data-parallel replicas (e.g., via summation) and returns only a disjoint $\frac{1}{N_d}$ shard of the reduced result to each replica.

Each replica subsequently performs an optimizer step using only its local shard of the optimizer states, corresponding to a $\frac{1}{N_d}$ fraction of the full optimizer state, where $N_d$ denotes the data-parallel degree. This yields an updated $\frac{1}{N_d}$ subset of the FP32 parameters, which are then cast to BF16. Finally, an *all-gather* operation is applied to the BF16 parameters to reconstruct the full set of updated model parameters on every replica. This additional all-gather is specific to ZeRO-style training and is not required in vanilla data parallelism.

![[Screenshot 2026-01-21 at 2.46.55 PM.png]]

It's like fragmenting it and the piecing it all together like puzzle pieces!

In ZeRO-1, we can also investigate how to efficiently overlap the newly added all-gather of BF16 parameters. There are two main strategies for this:

- **During the optimizer step:** We can initiate the all-gather immediately after the optimizer updates the first slice of the parameters. This allows the communication to potentially overlap with the updating of the other parameters.
- **During the forward pass:** We can overlap the all-gather of each layer’s parameters with the forward pass.

ZeRO-2 extends ZeRO-1 by sharding gradients across data-parallel ranks. Since each rank only updates a $\frac{1}{N_d}$ shard of the optimizer states, it only requires the corresponding $\frac{1}{N_d}$ shard of the gradients. Thus, during the backward pass, gradients are aggregated using a *reduce-scatter* operation instead of an *all-reduce*, reducing memory usage relative to ZeRO-1.

![[Screenshot 2026-01-21 at 4.22.21 PM.png]]

Thus, while ZeRO-2 significantly reduces memory consumption through gradient partitioning, it does not increase communication volume relative to standard data-parallel training.

![[Screenshot 2026-01-21 at 4.23.09 PM.png]]

In ZeRO-3, optimizer states, gradients, and model parameters are fully sharded across data-parallel replicas. Since no replica stores the complete set of parameters, parameters are all-gathered on demand during computation. During the forward pass, parameters for each layer are gathered just before use and immediately released afterward, while the backward pass follows the same process in reverse order, producing sharded gradients.

Compared to ZeRO-2, ZeRO-3 introduces additional communication overhead due to frequent parameter all-gathers, resulting in approximately $2 \cdot \text{num\_layers} - 1$ extra all-gather operations per training step. Each training step incurs three communication phases: an all-gather for parameters during the forward pass, an all-gather during the backward pass, and a reduce-scatter for gradients, yielding a total communication cost of $3\Psi$, compared to $2\Psi$ for ZeRO-2.

This overhead is mitigated through prefetching, which overlaps communication with computation by all-gathering parameters for the next layer while executing the current layer. Prefetching is effective as long as the data-parallel degree remains moderate (typically $N_d \lesssim 512$). In terms of memory, ZeRO-3 achieves maximal parameter sharding, reducing model-related memory usage proportionally to $\frac{1}{N_d}$, though activation memory remains unchanged and must be addressed separately via techniques such as activation checkpointing and gradient accumulation.

Data parallelism does not reduce activation memory per replica unless the per-replica microbatch size is reduced, in which case activation memory scales proportionally with the microbatch size.

Our next trick is called **Tensor Parallelism**

# Tensor Parallelism


