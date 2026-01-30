
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



