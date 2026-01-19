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

llama1: bs ~ 4 million tokens; 1.4 trillion total tokens
deepseekR1: bs ~ 60 million tokens; 14 trillion total tokens


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
g$$
Per layer:
- LayerNorm scale (γ): `h`
- LayerNorm bias (β): `h`

Memory requirements for the parameters and gradients are determined by multiplying the number of parameters by the number of bytes per parameter. Full precision (FP32) training, both parameters and gradients require 4 bytes while the optimizer, if we use Adam, requires the momentum and variance to be stored, adding another 8 bytes per parameter

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
Mixed precision doesn't directly reduce memory, it adds 4 bytes of data over full precision. It instead provides two massive advantages:
1. Compute the forward/backward passes in half precision
2. Allows us to use optimized lower precision operations on the GPU, which are faster
3. Reduces the activation memory requirements during the forward pass



  
