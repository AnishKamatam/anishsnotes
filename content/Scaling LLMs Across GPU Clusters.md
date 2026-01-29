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

---
# Activation Recomputation
Activation Recomputation is also known as gradient checkpointing. The general idea behind activation recomputation is to discard some activations during the forward pass to save memory and spend some extra compute to recompute these on the fly during the backward pass.
![[Screenshot 2026-01-19 at 3.08.38 PM.png]]
Instead of storing every single activation, we discard some of the activations, and then recompute on the fly. This adds computation, but saves memory.

There are a few strategies for selecting key activations to store:

- **Full**: Checkpoint activations at the transition point between each layer of the Transformer model. It requires a forward pass through each layer, essentially adding a full forward pass during the backward pass. Saves the most memory but is the most expensive one in terms of compute. It typically increases the compute cost and time by up to 30-40%


- **Selective**: Discard the attention computations. Focus on checkpointing the expensive feedforward computations. **70% activation memory reduction at a 2.7% compute cost**.

**Activation recomputation slightly increases the number of FLOPS due to recomputation, while it significantly reduces memory access overhead.**

This trade-off is particularly advantageous on hardware with limited high-speed memory, like GPUs, as accessing memory is typically slower than performing computations. Despite the additional operations involved, the overall effect is thus often faster computation, in addition to the much lower memory footprint.

---
# Gradient Accumulation
Gradient accumulation is a very straightforward method that consists of splitting a batch into micro-batches. Perform forward and backward passes successively on each micro-batch, compute the gradients, and then sum the gradients of all micro-batches before we perform optimization.
![[Screenshot 2026-01-19 at 3.41.54 PM.png]]
Gradient accumulation allows us to reduce activation memory by processing smaller micro-batches sequentially. This reduces stored activations and gradients since only one micro-batch's worth of activations needs to be kept in memory at a time.

However, gradient accumulation requires multiple consecutive forward/backward passes per optimization step, thereby increasing the compute overhead and slowing down training. Another approach is to perform the forward pass and backward pass of each microbatch in parallel without waiting for each batch. This leads us to ...

---
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

---
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

---
# Tensor Parallelism

While **ZeRO** effectively shards model parameters, gradients, and optimizer states, large-scale models often hit a "memory wall" when **activation memory** exceeds the available GPU budget.

**Tensor Parallelism (TP)** addresses this by sharding weights, gradients, optimizer states, _and_ activations across multiple devices. Unlike ZeRO, which requires gathering sharded data before computation, TP performs the computation on sharded data directly.
## The Mathematical Foundation

Tensor parallelism leverages the distributive properties of matrix multiplication ($A \times B$). To understand how it works, let's examine the two fundamental ways to shard a matrix product:
### 1. Column Parallelism

We can split matrix $B$ by columns. This allows each device to compute a portion of the output independently.

$$A \cdot B = A \cdot [B_1, B_2, \dots] = [AB_1, AB_2, \dots]$$
### 2. Row Parallelism

We can split matrix $A$ by columns and matrix $B$ by rows. The final result is the sum of the partial products.

$$A \cdot B = [A_1, A_2, \dots] \begin{bmatrix} B_1 \\ B_2 \\ \vdots \end{bmatrix} = \sum_{i=1}^{n} A_i B_i$$

---
## Application in Neural Networks
In the context of a Transformer or a standard feed-forward layer, we typically represent matrix multiplication as $X \times W$, where:
- **$X$ (Activations):** Represent the input values or hidden states.
- **$W$ (Weights):** Represent the learnable parameters of the Linear layer.
### How TP applies to $X \times W$:
1. **Column Parallel Approach:** We shard the weight matrix $W$ vertically. Each GPU receives the full input $X$ but only a portion of the weights ($W_{col}$), producing a shard of the output activations.
2. **Row Parallel Approach:** We shard the weight matrix $W$ horizontally. Each GPU receives only a portion of the input $X$ (sharded by columns) that corresponds to its weight shard ($W_{row}$), and the results are summed via an `AllReduce` operation.

> **Key Advantage:** By sharding the weights and the resulting output, the memory footprint of the activations is distributed across all participating GPUs, effectively breaking the activation memory bottleneck.

### Strategic Integration in Transformer Blocks

In a real-world Transformer, we combine these two methods within the MLP and Attention blocks to minimize communication overhead.

For the **MLP block**, the most efficient schema is a **Column-Linear layer followed by a Row-Linear layer**. This configuration is highly effective because it requires only one "exposed" synchronization point (the All-Reduce) at the very end of the block. The intermediate activations stay sharded, which prevents unnecessary communication.

The **Multi-Head Attention (MHA)** block follows a similar logic. We shard the Query (Q), Key (K), and Value (V) projections column-parallelly, allowing each GPU to independently handle a subset of attention heads. The output projection then acts as the row-linear layer to combine them. This naturally scales with the number of heads; for example, a model with 32 heads can theoretically scale to a TP degree of 32. However, for architectures like **Grouped Query Attention (GQA)**, where there are fewer Key/Value heads than Query heads (as seen in Llama-3), extra care is needed to keep the K/V heads synchronized across the TP ranks.

### Performance Trade-offs and the "NVLink Wall"

Tensor parallelism is not a "free" gain. Because communication primitives like All-Reduce sit directly in the critical path of the forward pass, they cannot be easily hidden or overlapped with computation like the communications in ZeRO-3.

While increasing the TP degree reduces the memory footprint per GPU—allowing us to fit massive 70B+ parameter models on a single node, it significantly impacts throughput. Within a single node (up to 8 GPUs), TP is highly performant because it utilizes high-speed **NVLink** interconnects. However, scaling TP across nodes (TP > 8) forces communication over standard network interfaces, leading to a steep decline in efficiency.

### Refinements and Limitations

Even with TP, some operations like **Layer Normalization** and **Dropout** traditionally require gathering full activations, which can limit potential memory savings. Interestingly, LayerNorm weights do not require gradient synchronization because every TP rank sees the same input after the All-Gather and naturally stays in sync. Conversely, Dropout requires explicit random seed synchronization to ensure deterministic behavior across the sharded ranks.

Ultimately, TP is a trade-off: you sacrifice a portion of your raw per-GPU throughput to unlock the ability to process much larger batch sizes and models that would otherwise be impossible to fit in memory.

---
# Sequence Parallelism
While Tensor Parallelism (TP) shards the heavy matrix multiplications (MLP and Attention) along the hidden dimension, it leaves certain operations, like **LayerNorm** and **Dropout** duplicated across GPUs. These operations require the full hidden dimension to compute statistics like mean and variance correctly. Consequently, in vanilla TP, every GPU still stores a full copy of the activations for these layers, creating a memory bottleneck as sequence lengths grow.

**Sequence Parallelism** solves this by sharding these "non-parallel" operations along the **sequence dimension** rather than the hidden dimension.
### The Mechanism: Shifting Dimensions

In a TP+SP configuration, the model alternates between two types of regions:
1. **SP Regions (LayerNorm/Dropout):** Activations are sharded along the sequence length ($s/TP$). Each GPU processes its own chunk of tokens independently.
2. **TP Regions (Linear Layers):** Activations must be gathered to the full sequence length ($s$) because the weights are sharded along the hidden dimension ($h/TP$).
The following diagram illustrates how a standard Transformer block is divided into these regions and the specific collective operations ($g$ and $g^*$) used to bridge them:

![[Pasted image 20260124105846.png]]

### Transitioning with Conjugate Pairs

To move between these regions efficiently, we use specific communication primitives that ensure correctness while keeping peak memory usage low:

- **From SP to TP ($g$ - All-Gather):** Before a linear layer, we use an **All-Gather** to combine the sequence shards ($s/TP \to s$). This restores the full sequence needed for column-linear layers to compute the hidden dimension.
    
- **From TP to SP ($g^*$ - Reduce-Scatter):** After a row-parallel linear layer, instead of a standard All-Reduce, we use a **Reduce-Scatter**. This sums the partial results for correctness while simultaneously scattering them back along the sequence dimension ($s \to s/TP$).
    

These operations are "conjugate pairs". In the forward pass, if the transition is an All-Gather, it becomes a Reduce-Scatter in the backward pass to synchronize gradients.

### Activation Memory Impact

The primary advantage of SP is the drastic reduction in peak activation memory. By ensuring that activations are _always_ sharded—either by the hidden dimension (in TP regions) or the sequence dimension (in SP regions)—the maximum activation size is reduced to $b \cdot s \cdot h / TP$.

This allows for significantly longer context windows. For a 70B model, adding SP to TP=16 can enable sequence lengths of 16k tokens that would otherwise trigger "Out of Memory" (OOM) errors.

### Communication Costs and Trade-offs

A common concern is whether SP adds more overhead than vanilla TP. Mathematically, one **All-Reduce** (used in TP) is equivalent in cost to one **All-Gather** plus one **Reduce-Scatter** (used in SP), so the total data transferred is identical.

However, these operations sit in the **critical path** of the forward pass and cannot be easily overlapped with computation. The following profile shows how these communication steps (AG and RS) create synchronization points that the GPU must wait for:

![[Screenshot 2026-01-24 at 10.59.28 AM.png]]

**Enter TP (Column-Linear Layers)**
- **TP only**
  - Hidden dimension ($h$): sharded
  - Sequence length ($s$): full
- **TP + SP**
  - Hidden dimension ($h$): sharded
  - Sequence length ($s$): All-Gathered to full

**Exit TP (Row-Linear Layers)**
- **TP only**
  - Hidden dimension ($h$): full (via All-Reduce)
  - Sequence length ($s$): full
- **TP + SP**
  - Hidden dimension ($h$): full
  - Sequence length ($s$): Reduce-Scattered to sharded

**SP Region (LayerNorm / Dropout / Residuals)**
- **TP only**
  - Hidden dimension ($h$): full
  - Sequence length ($s$): full
- **TP + SP**
  - Hidden dimension ($h$): full
  - Sequence length ($s$): sharded
### Implementation Nuance

- **LayerNorm Gradients:** Because each TP rank in the SP region operates on different portions of the sequence, their gradients will differ. To keep weights synchronized, we must All-Reduce their gradients during the backward pass.

- **Performance Wall:** Just like vanilla TP, TP+SP is most effective within a single node (TP $\leq$ 8). Moving across nodes (e.g., TP=16) results in a massive throughput drop as communication shifts from NVLink to inter-node networking.

---
# Context Parallelism

When scaling models to handle massive sequence lengths, such as 128k or more tokens, even the combination of Tensor Parallelism and Sequence Parallelism hits a limit. This occurs because the full sequence must still be processed when inside a TP region, and even with activation recomputation, memory usage at layer boundaries continues to scale linearly with the sequence length. Context Parallelism addresses this by sharding the sequence dimension across the entire model, including the regions typically handled by Tensor Parallelism.

The core idea of Context Parallelism is to split the input along the sequence dimension for the full duration of the model computation. For most modules like the MLP and LayerNorm, this process is intuitive because tokens are processed independently. Splitting the sequence for these layers does not require expensive communication like TP because only the inputs are split rather than the weight matrices. Just as with Data Parallelism, an all-reduce operation is initiated after computing gradients to synchronize them across the CP group.

The primary challenge lies in the attention blocks where each token needs to access key and value pairs from other sequence tokens to compute correctly. Because Context Parallelism distributes these inputs across GPUs, the attention module requires full communication between devices to exchange the necessary data. To handle this efficiently without high costs, a technique called Ring Attention is used. This enables the communication of key and value pairs by passing them in a circular fashion between GPUs to maintain performance while processing massive context windows

### Ring Attention

In this implementation of the attention mechanism, each GPU first initiates an asynchronous communication operation to send its key and value pairs to other GPUs. While waiting for the other GPUs' data, it computes the attention score for the portion of the data it already has in memory. Ideally, the next key and value pair is received from another GPU before this computation finishes, allowing the GPU to start the next round of computation immediately after it finishes its first computation.

For a setup with four GPUs and an input of four tokens, the input sequence is split evenly along the sequence dimension so each GPU has one token along with its corresponding Q, K, and V values. The attention calculation takes four time steps to complete, and at each time step, each GPU performs three successive operations:
Here's what that looks like: 

![[Pasted image 20260124131724.png]]
- The GPU sends current keys and values to the next machine in a non-blocking manner.
- The GPU locally computes the attention score on the current keys and values, which involves performing $Softmax(\frac{QK^T}{\sqrt{d}})V$.
- The GPU waits to receive keys and values from the previous GPU and then circles back to the first step.

A significant problem with a naive implementation of Ring Attention is the strong imbalance between GPUs caused by the shape of the causal attention matrix. Softmax is computed row-wise, meaning a GPU can only compute it once it has received all the tokens of a row. In this scenario, the first GPU can compute immediately as it starts with the necessary tokens, but the second GPU must wait for the second round to receive enough information, resulting in the first GPU performing much less work than the others.

### Zig-Zag Ring Attention

To resolve the computational imbalance, input sequences can be distributed using **Zig-Zag Attention**. Instead of assigning tokens in a purely sequential manner, this approach mixes the ordering so each GPU receives a combination of early and late tokens. This ensures that the attention mask reflects an even distribution of work across all GPUs. Under this arrangement, every GPU eventually requires information from all others to complete its rows, but the workload remains balanced.

There are two primary ways to manage the overlap of computation and communication within this framework:
**1. All-gather Implementation**
- All GPUs simultaneously gather the complete key/value pairs from all other GPUs.
- This requires more temporary memory because each GPU must store all K/V pairs at once.
- While communication happens in a single step, the memory overhead is higher.
**2. All-to-all (Ring) Implementation**
- GPUs exchange K/V pairs in a ring-like pattern, one chunk at a time.
- This is more memory-efficient as each GPU only needs to store one additional chunk temporarily.
- Communication is spread out and overlapped with computation, though multiple communication steps introduce additional base latency

### Combining Sequence Parallelism and Context Parallelism

When combining Sequence Parallelism (SP) with Context Parallelism (Ring Attention), there is an important constraint: $\text{sp\_size} \times \text{ring\_size} = \text{num\_gpus}$, if either of them are set. This ensures that the total number of GPUs is properly partitioned between the two parallelism strategies.

While Context Parallelism tames the activation explosion associated with long sequences, Tensor Parallelism remains difficult to scale across nodes. If model weights cannot fit on a single node, **Pipeline Parallelism** serves as the next degree of parallelism to resolve the bottleneck.

---
# Pipeline Parallelism

While Sequence and Context Parallelism address memory issues stemming from long sequences, they do not resolve the problem of model size itself. For large models with 70B or more parameters, the size of the weights alone can push past the limits of a single node. **Pipeline Parallelism (PP)** resolves this by splitting the model's layers across multiple GPUs, allowing each device to store and process only a portion of the model.
### Pipeline Mechanism and Activation Memory
Conceptually, PP involves passing activation tensors sequentially between GPUs in a "pipeline". However, a key technical note is that while model parameters are distributed, activation memory remains roughly the same on each GPU. This happens because each GPU handles a fraction of the layers ($1/PP$) but must process multiple micro-batches ($PP$) before starting the first backward pass, resulting in an activation requirement of approximately $PP \times (\text{activations}/PP) \approx \text{activations}$.
### Communication and Efficiency
PP offers a distinct advantage in communication bandwidth because it only sends moderate-sized activations at specific locations along the model depth, rather than communicating several times within each layer as seen in Tensor Parallelism. Despite this simplicity, the sequential nature of PP introduces a major efficiency challenge known as the **pipeline bubble**.

![[Screenshot 2026-01-24 at 2.04.34 PM.png]]
### The Pipeline Bubble

In a naive forward and backward pass, GPUs often sit idle while waiting for data from the previous or subsequent stage. This idle time, indicated as the "bubble," directly reduces throughput. We can quantify this inefficiency by comparing the idle time to the ideal computation time:

- **Variables**: Let $t_f$ and $t_b$ represent the time for forward and backward passes for one micro-batch in one stage.
- **Ideal Time**: The ideal total time would be $t_{id} = t_f + t_b$.
- **Bubble Time**: The additional time lost to the bubble is $t_{pb} = (p-1) \times (t_f + t_b)$, where $p$ is the degree of pipeline parallelism.
- **Bubble Ratio**: The ratio of lost time over ideal time simplifies to $r_{bubble} = p - 1$.

As the degree of pipeline parallelism increases, the bubble time grows and overall GPU utilization drops significantly in a naive implementation.

To address the inherent inefficiencies of Pipeline Parallelism (PP), we can optimize model training by splitting batches into smaller **micro-batches**. This technique allows multiple GPUs to work on different segments of a training step simultaneously, significantly improving device utilization.

### All Forward, All Backward (AFAB) Schedule

The **All Forward, All Backward (AFAB)** schedule is one of the simplest implementations of pipeline parallelism. In this setup, each device completes its forward passes for every micro-batch before moving on to any backward passes. While the AFAB schedule maintains a clear sequential organization in the training code making it easier to implement it still faces challenges with the **pipeline bubble**.

![[Screenshot 2026-01-24 at 2.08.54 PM.png]]

We can quantify the efficiency of this pipeline setup by calculating the ratio of the additional bubble time over the ideal time:

- **Variables**: Let $t_f$ and $t_b$ represent the times for the forward and backward passes for one micro-batch in one stage.
- **Ideal Time**: To process $m$ micro-batches, the ideal time is $t_{id} = m \times (t_f + t_b)$.
- **Bubble Time**: The time each GPU spends idling while others compute is $t_{pb} = (p-1) \times (t_f + t_b)$, where $p$ is the number of pipeline stages (GPUs).
- **Bubble Ratio**: The efficiency loss is expressed as $r_{bubble} = \frac{p - 1}{m}$.

By increasing the number of micro-batches ($m$), we can fight some of the inefficiencies of pipeline stages, effectively reducing the relative size of the bubble and improving GPU throughput.

### The Challenge of Activation Memory

A critical drawback of the AFAB approach is the **memory explosion** caused by stored activations. Because every forward pass must be completed before any backward pass begins, the GPU must hold the activations for _all_ micro-batches in memory simultaneously.

Since each GPU handles $1/PP$ of the layers but needs to process $PP$ micro-batches before the first backward pass, it ends up storing $PP \times (\text{activations}/PP)$, which means the activation memory requirement remains roughly the same as it would be without pipeline parallelism. To mitigate this issue, we must look at schedules that begin the backward pass while the forward computation is still ongoing, allowing the system to drop activations as soon as they are utilized for gradient calculation.

To address the activation memory explosion inherent in the AFAB schedule, we can transition to a more efficient strategy known as **One Forward, One Backward (1F1B)**. This schedule optimizes memory by beginning the backward pass as soon as the first forward pass completes for a given stage.

### 1F1B Schedule and Memory Efficiency

The core principle of 1F1B is the interleaved execution of forward and backward passes. In the "steady state" of this schedule, each GPU alternately performs one forward pass and one backward pass.

![[Pasted image 20260124163927.png]]

While the size of the pipeline bubble remains mathematically identical to the AFAB schedule, 1F1B offers a critical memory advantage:

- **Memory Footprint**: In 1F1B, a GPU only needs to store activations for $p$ micro-batches (where $p$ is the degree of pipeline parallelism).
- **Comparison to AFAB**: This is a significant reduction from the AFAB schedule, which requires storing activations for all $m$ micro-batches ($m \gg p$).
- **Scaling Potential**: Because the per-GPU memory burden is lower, 1F1B allows us to use a much higher number of micro-batches, which in turn reduces the relative size of the pipeline bubble.
### Implementation Complexity and Challenges
Implementing 1F1B is considerably more complex than naive pipeline schedules. Because forward and backward passes are interleaved and performed in parallel across different devices, the training loop can no longer be a simple, central sequential loop. Each device must independently manage its own switch between forward and backward tasks. This often requires extensive modifications to both the training framework and the underlying model code.
### Performance and Inter-node Scaling
Benchmarks demonstrate that 1F1B is particularly effective for multi-node training:
- **Bubble Impact**: When the number of micro-batches is small ($m \approx p-1$), performance is low and degrades as the PP degree increases.
- **Large Batch Scaling**: Using many more micro-batches than pipeline stages ($m \gg p-1$) significantly improves throughput.
- **Cross-node Resilience**: Unlike Tensor Parallelism, which suffers heavy performance hits when crossing between nodes, 1F1B scales well across the network. For example, scaling from 8 GPUs (one node) to 16 GPUs (two nodes) may only see a 14% performance drop, compared to over 40% for Tensor Parallelism.

This resilience to lower-bandwidth inter-node connections makes Pipeline Parallelism with 1F1B one of the most attractive strategies for distributing massive models across large clusters.

While the 1F1B schedule improves memory usage, it does not reduce the idle pipeline bubble. To further minimize this wasted time, we can implement **Interleaved Stages**, which involves slicing model layers across GPUs in a non-sequential manner. Instead of hosting layers 1-4 on GPU 1 and 5-8 on GPU 2, we might distribute layers such that each GPU handles multiple "virtual stages" throughout the model's depth.

### The Looping Pipeline

This configuration creates a "looping pipeline" where a micro-batch moves between GPUs multiple times to complete a single forward or backward pass. While this increases the number of communication operations, it allows for much tighter interleaving of forward and backward steps.

![[Screenshot 2026-01-24 at 6.02.17 PM.png]]

The efficiency gain can be quantified by looking at the reduction in bubble time, where v is the number of virtual stages (model chunks) per GPU:

-  **Bubble Time**: $t_{pb} = \frac{(p-1) \times (t_f + t_b)}{v}$.
- **Bubble Ratio**: $r_{bubble} = \frac{p-1}{v \times m}$.

By adding interleaved stages, the bubble is decreased by a factor of v, though this is a trade-off as total communication also increases by that same factor.
### Scheduling Strategies

Interleaved pipelines introduce significant scheduling complexity. At any given moment, a GPU must decide between two priorities:

- **Depth-First**: Prioritizes earlier micro-batches in later layers to finish the full pass as fast as possible.
- **Breadth-First**: Prioritizes later micro-batches in earlier layers to keep the pipeline filled.

Advanced models like Llama 3.1 utilize 1F1B setups with interleaved stages and tunable priority settings between these two approaches.


While the interleaved 1F1B schedule significantly improves efficiency, even more sophisticated methods have recently been proposed to reach a "zero bubble" regime. These techniques, such as the **DualPipe** implementation used in DeepSeek-V3 and R1, aim to achieve near-zero all-to-all communication overhead by splitting model operations into even finer-grained components.

### Decomposing the Backward Pass

The core insight behind these "zero bubble" schedules is that a standard backward pass through a matrix multiplication actually consists of two separate operations:
- **B (Backward for Inputs)**: Calculates the gradient with respect to the input, which is necessary for the backward pass of preceding layers.
- **W (Backward for Weights)**: Calculates the gradient with respect to the weights, which only needs to be completed before the final optimizer step.

![[Pasted image 20260125112035.png]]

Because the weight gradient ($W$) is not required for the backward pass of lower layers, it can be flexibly scheduled at any point after the corresponding input gradient ($B$) of the same stage. This flexibility allows for the strategic placement of $W$ operations to fill the idle gaps that would otherwise form pipeline bubbles.

### DualPipe and Bidirectional Streams

DeepSeek’s **DualPipe** extends this decomposition further by introducing two simultaneous processing streams that propagate from both ends of the pipeline. These bidirectional streams are interleaved to further minimize idle time, creating a highly efficient but extremely complex scheduling graph.
![[Pasted image 20260125112046.png]]
Fully optimizing such schedules typically involves precisely measuring the duration of each fine-grained operation and solving an Integer Linear Programming (ILP) problem to minimize bubble time. While these implementations are too complex for simple code snippets, they represent the current frontier in eliminating the efficiency bottlenecks of pipeline parallelism.

