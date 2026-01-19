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

 For short sequences (or small batch sizes), memory usage for activations is almost negligible, but from around 2-4k tokens they start to take up a significant amount of memory, while usage for parameters, gradients, and optimizer states  is roughly independent of the sequence length and batch size.

# Activation Recomputation
Activation Recomputation is also known as gradient checkpointing. The general idea behind activation recomputation is to discard some activations during the forward pass to save memory and spend some extra compute to recompute these on the fly during the backward pass.
![[Screenshot 2026-01-19 at 3.08.38 PM.png]]
Instead of storing every single activation, we discard some of the activations, and then recompute on the fly. This adds computation, but saves memory.

There are a few strategies for selecting key activations to store:

- **Full**: Checkpoint activations at the transition point between each layer of the Transformer model.Requires a forward pass through each layer, essentially adding a full forward pass during the backward pass. Saves the most memory but is the most expensive one in terms of compute. It typically increases the compute cost and time by up to 30-40%


- **Selective**: Discard the attention computations. Focus on checkpointing the expensive feedforward computations. **70% activation memory reduction at a 2.7% compute cost**.

**Activation recomputation slightly increases the number of FLOPS due to recomputation, while it significantly reduces memory access overhead.**

This trade-off is particularly advantageous on hardware with limited high-speed memory, like GPUs, as accessing memory is typically slower than performing computations. Despite the additional operations involved, the overall effect is thus often faster computation, in addition to the much lower memory footprint.


# Gradient Accumulation
Gradient accumulation is a very straightforward method that consists of splitting a batch into micro-batches. Perform forward and backward passes successively on each micro-batch, compute the gradients, and then sum the gradients of all micro-batches before we perform optimization.
![[Screenshot 2026-01-19 at 3.41.54 PM.png]]
Gradient accumulation allows us to reduce activation memory by processing smaller micro-batches sequentially. This reduces stored activations and gradients since only one micro-batch's worth of activations needs to be kept in memory at a time.

However, gradient accumulation requires multiple consecutive forward/backward passes per optimization step, thereby increasing the compute overhead and slowing down training. Another thing, we can also do is we can do the forward pass and backward pass of each microbatch in parallel without waiting for each batch. Leading us to ...

# Data Parallelism

