# GPT-2 Text Generation

**Difficulty**: Hard  
**Topics**: GPT-2, Transformers, NLP, Deep Learning, Text Generation, Language Models  
**Problem Link**: [https://www.deep-ml.com/problems/88](https://www.deep-ml.com/problems/88?from=Deep%20Learning)

## Problem Description

This problem involves implementing GPT-2 text generation from scratch, focusing on understanding the core components of transformer-based language models including token embeddings, positional encodings, and layer normalization.

## My Approach

My implementation focuses on the fundamental building blocks of GPT-2 text generation:

1. **Token and Positional Embeddings**: Combined word token embeddings (`wte`) with positional encodings (`wpe`) to give the model both semantic and positional information about each token.

2. **Layer Normalization**: Implemented layer normalization carefully, ensuring it's applied per row (per token) rather than across the entire matrix. This is crucial for maintaining the correct statistical properties of each token's representation.

3. **Autoregressive Generation**: Used an iterative approach where the model generates one token at a time, appending each new token to the sequence and using it as context for the next prediction.

4. **Greedy Decoding**: Selected the token with the highest probability (argmax of logits) at each step for deterministic output.

## Key Implementation Details

- **Layer Normalization**: Applied per token with careful attention to the axis of operations
- **Randomness Control**: Set random seed for reproducible testing outputs
- **Token Management**: Tracked initial prompt tokens separately to return only generated text
- **Logit Calculation**: Used matrix multiplication with transposed embedding weights to project back to vocabulary space

## Solution

See `solution.py` for the complete implementation.

## Key Insights

- Layer normalization must be applied correctly per row to maintain token-level statistics
- The autoregressive nature of GPT-2 means each token generation depends on all previous tokens
- Positional encodings are essential for the model to understand token order
- Greedy decoding is simple but effective for basic text generation

## Challenges Faced

The main challenge was ensuring layer normalization was applied correctly. It's easy to accidentally normalize across the wrong dimension, which would mix statistics across different tokens and break the model's behavior.

## Related Resources

- [Andrej Karpathy's GPT Video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1193s) - Excellent deep dive into GPT architecture
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---
