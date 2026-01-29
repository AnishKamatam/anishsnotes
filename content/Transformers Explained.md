
# What is a Transformer?

A **Transformer** is a neural network architecture that has fundamentally reshaped modern artificial intelligence. First introduced in the 2017 paper _“Attention Is All You Need”_, Transformers have become the dominant backbone of today’s most powerful AI systems. They power large language models such as OpenAI’s GPT series, Meta’s LLaMA, and Google’s Gemini, and have also proven effective far beyond text, spanning audio generation, computer vision, protein structure prediction, and even game playing.

At their core, text-generative Transformer models are trained to perform **next-token prediction**. Given a sequence of tokens, which can be words or sub-words, the model estimates the probability distribution over what token should come next. By repeatedly predicting one token at a time, Transformers are able to generate coherent paragraphs, write code, answer questions, and carry on conversations.

What makes Transformers especially powerful is their use of the **self-attention mechanism**. Unlike earlier architectures such as recurrent neural networks, which process tokens sequentially, self-attention allows a Transformer to consider the entire input sequence at once. Each token can selectively attend to every other token, enabling the model to capture long-range dependencies, contextual relationships, and subtle patterns in language far more effectively.

### Transformer Architecture
Every text-generative Transformer is built from three core components that work together to turn text into predictions.
#### 1. Tokenization and Embeddings
The Transformer does not operate directly on raw text. Instead, input text is first broken into smaller units called **tokens**, which can represent full words, subwords, or even individual characters. Each token is then mapped to a dense numerical vector known as an **embedding**.

These embeddings capture semantic information about the tokens. Words with similar meanings tend to have embeddings that are close together in this high-dimensional space. The resulting sequence of embeddings serves as the input to the Transformer model.

---
#### 2. Transformer Blocks
The **Transformer block** is the fundamental computational unit of the model. A Transformer consists of many such blocks stacked on top of each other, with each block progressively refining the token representations.

Each Transformer block contains two main subcomponents.

**Self-Attention Layer**
The self-attention mechanism is the core innovation of the Transformer architecture. It allows each token to dynamically exchange information with every other token in the sequence. Through self-attention, the model learns which tokens are relevant to each other and how strongly they should influence one another. This enables the Transformer to capture long-range dependencies and contextual relationships that are difficult for earlier architectures to model.

**Feed-Forward Network (MLP)**
Following the attention layer, each token is passed through a feed-forward neural network, often called an MLP. This network operates on each token independently. While attention focuses on routing information between tokens, the MLP focuses on **transforming and refining the representation of each token itself**. Together, these two components allow the model to both share information across the sequence and deepen its internal understanding of each token.

---
#### 3. Output Projection and Probabilities

After passing through the stack of Transformer blocks, the model produces a final set of hidden representations, one for each token. These representations are then passed through a linear projection followed by a softmax function to produce a probability distribution over the vocabulary.

This probability distribution represents the model’s belief about which token is most likely to come next, enabling autoregressive text generation one token at a time.

---

### Why this structure works

- **Embeddings** convert language into a form the model can process.
- **Attention layers** let tokens share information and build context.
- **MLP layers** refine token representations individually.
- **Output layers** translate learned representations into concrete predictions.

This modular design is what allows Transformers to scale efficiently and generalize across many different tasks and domains.