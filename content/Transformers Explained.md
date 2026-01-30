
# What is a Transformer?

A **Transformer** is a neural network architecture that has fundamentally reshaped modern artificial intelligence. First introduced in the 2017 paper _“Attention Is All You Need”_, Transformers have become the dominant backbone of today’s most powerful AI systems. They power large language models such as OpenAI’s GPT series, Meta’s LLaMA, and Google’s Gemini, and have also proven effective far beyond text, spanning audio generation, computer vision, protein structure prediction, and even game playing.

![[Pasted image 20260129151320.png]]
At their core, text-generative Transformer models are trained to perform **next-token prediction**. Given a sequence of tokens, which can be words or sub-words, the model estimates the probability distribution over what token should come next. By repeatedly predicting one token at a time, Transformers are able to generate coherent paragraphs, write code, answer questions, and carry on conversations.

What makes Transformers especially powerful is their use of the **self-attention mechanism**. Unlike earlier architectures such as recurrent neural networks, which process tokens sequentially, self-attention allows a Transformer to consider the entire input sequence at once. Each token can selectively attend to every other token, enabling the model to capture long-range dependencies, contextual relationships, and subtle patterns in language far more effectively.

### Transformer Architecture
Every text-generative Transformer is built from three core components that work together to turn text into predictions.
#### 1. Tokenization and Embeddings
The Transformer does not operate directly on raw text. Instead, input text is first broken into smaller units called **tokens**, which can represent full words, subwords, or even individual characters. Each token is then mapped to a dense numerical vector known as an **embedding**. These embeddings capture semantic information about the tokens. Words with similar meanings tend to have embeddings that are close together in this high-dimensional space. The resulting sequence of embeddings serves as the input to the Transformer model.

---
#### 2. Transformer Blocks
The **Transformer block** is the fundamental computational unit of the model. A Transformer consists of many such blocks stacked on top of each other, with each block progressively refining the token representations.
Each Transformer block contains two main subcomponents.
- **Self-Attention Layer**
	The self-attention mechanism is the core innovation of the Transformer architecture. It allows each token to dynamically exchange information with every other token in the sequence. Through self-attention, the model learns which tokens are relevant to each other and how strongly they should influence one another. This enables the Transformer to capture long-range dependencies and contextual relationships that are difficult for earlier architectures to model.
- **Feed-Forward Network (MLP)**
	Following the attention layer, each token is passed through a feed-forward neural network, often called an MLP. This network operates on each token independently. While attention focuses on routing information between tokens, the MLP focuses on **transforming and refining the representation of each token itself**. Together, these two components allow the model to both share information across the sequence and deepen its internal understanding of each token.

---
#### 3. Output Projection and Probabilities

After passing through the stack of Transformer blocks, the model produces a final set of hidden representations, one for each token. These representations are then passed through a linear projection followed by a softmax function to produce a probability distribution over the vocabulary. This probability distribution represents the model’s belief about which token is most likely to come next, enabling autoregressive text generation one token at a time.

---

### Why this structure works
- **Embeddings** convert language into a form the model can process.
- **Attention layers** let tokens share information and build context.
- **MLP layers** refine token representations individually.
- **Output layers** translate learned representations into concrete predictions.

This modular design is what allows Transformers to scale efficiently and generalize across many different tasks and domains.

# The Embedding Layer

To convert a text prompt into a format a model can process, it must pass through the **Embedding layer**. This transforms the text into a numerical representation that the model can work with through a specific four-step sequence.
#### 1. Tokenization
Tokenization is the process of breaking down input text into smaller, manageable pieces called tokens. These tokens can represent whole words or subwords. For example, in a prompt like "AI research accelerates progress":
- Common words like **"research"** and **"progress"** might correspond to unique tokens.
- More complex words like **"accelerates"** might be split into multiple subword tokens.
- The full vocabulary of unique tokens is determined before the model is trained.
#### 2. Token Embedding
Once the input is split into tokens with distinct IDs, the model obtains their vector representations:
- Each token in the vocabulary is represented as a high-dimensional vector, the size of which depends on the specific model architecture.
- These embedding vectors are stored in an extensive matrix that allows the model to assign semantic meaning to each token.
- Tokens with similar usage or meaning in language are placed close together in this high-dimensional space, while dissimilar tokens are farther apart.
#### 3. Positional Encoding
The embedding layer also encodes information about each token's position in the input prompt. Because Transformer models do not inherently understand the order of a sequence, they require a specific method for positional encoding. Different models use various techniques, such as training a positional encoding matrix from scratch and integrating it directly into the training process.
#### 4. Final Embedding
Finally, the token embeddings and the positional encodings are summed to create the final representation. This combined representation captures both the semantic meaning of the tokens and their specific position in the input sequence.

# The Transformer Block
The **Transformer block** is the fundamental processing unit of the architecture, consisting of a **multi-head self-attention** mechanism and a **Multi-Layer Perceptron (MLP)** layer. Most models stack these blocks sequentially, allowing token representations to evolve and build an increasingly complex understanding of the input as they pass from the first block to the last.

---
### Multi-Head Self-Attention

>Self-attention allows the model to map relationships between tokens in a sequence, ensuring that each token’s final representation is influenced by every other token. By using multiple **attention heads**, the model can track these relationships from various perspectives simultaneously—such as focusing on grammatical structure in one head and broad thematic context in another.
#### 1. Tokenization
Each token's input embedding is transformed into three distinct vectors, **Query (Q)**, **Key (K)**, and **Value (V)** by multiplying it with learned weight matrices. You can think of these through a search engine analogy:
- **Query (Q):** The search term you type; represents the information a token is currently "looking for".
- **Key (K):** The titles of the search results; represents the information a token "offers" to other queries.
- **Value (V):** The actual content of the webpages; the data we extract once we find a match between a Query and a Key.

$$
QKV_{ij} = \left(\sum_{d=1}^{hidden\_size} Embedding_{i,d} \cdot Weights_{d,j}\right) + Bias_j
$$

#### 2. Multi-Head Splitting
The Q, K, and V vectors are split into several heads. Each head independently processes a different segment of the embedding, allowing the model to learn diverse linguistic features in parallel.
#### 3. Masked Self-Attention
Within each head, the model calculates attention while strictly following a causal constraint:
- **Dot Product:** Multiplying $Q$ and $K$ produces a square matrix of **attention scores**, reflecting the raw relationship between all tokens.
- **Scaling & Masking:** Scores are scaled to maintain stability, and a mask is applied to the upper triangle of the matrix. This ensures the model cannot "peek" into future tokens while predicting the next one.
- **Softmax & Dropout:** The scores are converted into probabilities via **Softmax**, so each row sums to 1.0, indicating exactly how much weight to give to preceding tokens.
#### 4. Output and Concatenation
Finally, the model multiplies these probabilities by the **Value (V)** matrix to produce the head's output. The outputs from all attention heads are concatenated and passed through a linear projection to be sent to the next part of the block.

# The Multi-Layer Perceptron (MLP)
After the multi-head self-attention mechanism captures relationships between tokens, the concatenated outputs are passed into the **Multi-Layer Perceptron (MLP)** layer. While attention integrates information _across_ tokens, the MLP processes each token **independently**, mapping their representations into a new space to enrich the model's overall capacity.
The MLP block typically follows a specific structure:

![[Pasted image 20260129161642.png]]

- **Expansion**: A linear transformation expands the input dimensionality four-fold (e.g., from a size of 768 to 3072). This projects the token into a higher-dimensional space to capture complex patterns.
- **Activation**: A non-linear activation function, such as **GELU**, is applied between the linear layers.
- **Projection**: A second linear transformation reduces the dimensionality back to the original size, retaining the useful non-linear transformations while making the representation manageable for the next block.
---
#### Output Probabilities and Token Prediction
Once the data has been processed through all sequential Transformer blocks, it reaches the final output stage to determine the next token in the sequence.
#### 1. Logit Generation
The final representations are projected into a massive dimensional space corresponding to the size of the vocabulary. Each token in the vocabulary is assigned a raw score called a **logit**.
#### 2. Softmax Transformation
The model applies the **softmax** function to convert these logits into a probability distribution where all values sum to one. This allows for the ranking and sampling of the next token based on its likelihood.
#### 3. Refined Sampling
The generation process is controlled by several key hyperparameters that balance determinism and "creativity":
- **Temperature**: The logits are divided by this value before softmax.
	- **$T = 1$**: No effect on the distribution.
	- **$T < 1$**: Makes the model more confident and deterministic (sharper distribution).    
	- **$T > 1$**: Creates a softer distribution, allowing for more randomness and diversity. 
- **Top-k Sampling**: Limits the candidates to the $k$ tokens with the highest probabilities.
- **Top-p (Nucleus) Sampling**: Selects the smallest set of tokens whose cumulative probability exceeds a threshold $p$.


### Auxiliary Architectural Features

Several auxiliary features enhance the performance of Transformer models. While they are not the primary mechanisms for processing tokens, they are crucial for stabilizing the training phase and ensuring the model generalizes well to new data.

---
### Layer Normalization

Layer Normalization stabilizes the training process and improves convergence by ensuring the mean and variance of activations remain consistent across features. This helps mitigate internal covariate shift, allowing the model to learn more effectively and reducing its sensitivity to initial weights.

- In each Transformer block, Layer Normalization is applied twice.
- It is positioned once before the self-attention mechanism.
- It is positioned a second time before the MLP layer.
### Dropout
Dropout is a regularization technique used during training to prevent overfitting.
- It works by randomly setting a fraction of model weights to zero, which forces the network to learn more robust features rather than depending on specific neurons.
- This technique helps the network generalize better to unseen data.
- During model inference, dropout is deactivated, effectively utilizing an ensemble of trained subnetworks for better performance.
### Residual Connections
Originally introduced in 2015, residual connections allow for the training of very deep neural networks by providing "shortcuts" that bypass layers. The input of a layer is added directly to its output.
- This mechanism helps solve the vanishing gradient problem, ensuring that earlier layers receive sufficient updates during backpropagation.
- Within each Transformer block, residual connections are utilized twice: once before the MLP and once after.
- This architecture allows gradients to flow more easily through the stacked blocks of the model.


### The Unified Transformer Workflow
The Transformer represents a shift from sequential processing to a massive, parallelizable architecture driven by **contextual relevance**. By combining semantic embeddings, dynamic attention, and independent refinement layers, it creates a system capable of both understanding intricate language nuances and generating highly coherent content.
The journey of a prompt through this architecture follows a precise, cyclical lifecycle:

---
### The Integrated Data Flow
1. **Preparation (Embedding Layer)**: Raw text is transformed into a rich numerical landscape. **Tokenization** breaks the string into IDs, **Token Embeddings** map those IDs to semantic vectors, and **Positional Encodings** inject the "where" into the "what." This ensures the model starts with a representation that captures both meaning and order.
2. **Synthesis (The Attention Head)**: Inside the first block, the **Self-Attention** mechanism creates a dialogue between tokens. Using **Queries, Keys, and Values**, each token determines which other parts of the sentence are most relevant to its own meaning. This is where the model resolves ambiguities—identifying, for instance, that "it" refers to "the ball" and not "the table."
3. **Refinement (The MLP & Residuals)**: The **MLP** takes the context-aware tokens and processes them independently to deepen their individual representations. All the while, **Residual Connections** and **Layer Normalization** act as a high-speed infrastructure, preventing the signals from degrading as they move through dozens of sequential blocks.
4. **Prediction (Output Head)**: The final representation is projected back into the vast space of the vocabulary. The **Softmax** function turns raw **logits** into a clear probability map. Through **Sampling** (governed by Temperature, Top-k, or Top-p), the model selects a single token and appends it to the input, restarting the entire cycle for the next word.