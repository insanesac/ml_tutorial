# LLM / Transformer Interview Roadmap

Use this as a running checklist. Check off topics as you can explain + implement them confidently.

---

## Foundations

- [ ] Tokenization
- [ ] Token IDs
- [ ] Embeddings
- [ ] Positional Encoding

---

## Attention Mechanism

- [ ] Q, K, V — what they represent
- [ ] Attention Scores: `Q · Kᵀ`
- [ ] Softmax in Attention
- [ ] Attention Weights
- [ ] Value Aggregation: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
- [ ] Multi-Head Attention

---

## Core Transformer Architecture

- [ ] Residual Connections
- [ ] Layer Normalization
- [ ] Feed-Forward Network (MLP/FFN)
- [ ] Full Transformer Block Assembly
- [ ] Masked Self-Attention (causal)

---

## Training

- [ ] Next Token Prediction objective
- [ ] Cross Entropy Loss
- [ ] Backpropagation through Transformers
- [ ] Training vs Inference differences

---

## Scaling & Inference

- [ ] Context Window
- [ ] O(N²) Attention Complexity
- [ ] KV Cache
- [ ] Why Larger Models Generalize Better

---

## LLM Systems Topics

- [ ] RAG — architecture + when to use
- [ ] Embeddings for Retrieval
- [ ] Vector Databases (FAISS, HNSW)
- [ ] Fine-Tuning overview
- [ ] LoRA — motivation + implementation
- [ ] Quantization — FP32 / FP16 / INT8 / INT4
- [ ] Hallucinations — causes + mitigations
- [ ] Prompt Engineering patterns
- [ ] Agents vs Workflows

---

## Coding Practice

- [ ] Implement Softmax (numerically stable)
- [ ] Implement Cross Entropy
- [ ] Implement Attention from Scratch
- [ ] Implement Embedding Lookup
- [ ] Implement Simple Transformer Block
- [ ] KV Cache (explain or implement)
- [ ] Complexity Analysis Questions

---

## L5 Discussion Readiness

For each topic below, be ready to discuss tradeoffs and design decisions at system level:

- How would you serve a 70B model with limited GPU memory?
- How does KV Cache change memory vs latency?
- When would you use LoRA vs full fine-tuning?
- How would you evaluate retrieval quality in a RAG system?
- What causes hallucinations and how do you reduce them?
