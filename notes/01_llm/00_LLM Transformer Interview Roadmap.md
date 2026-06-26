# LLM / Transformer Interview Roadmap

Use this as a running checklist. Check off topics as you can explain + implement them confidently.

**For L5**: You must not only know each topic but be able to discuss tradeoffs, design systems around them, and dive deeper on any follow-up question. For every item, ask yourself: "Could I whiteboard this, explain the math, and discuss system-level implications?"

---

## Foundations

- [ ] Tokenization (BPE, WordPiece, SentencePiece, Unigram)
- [ ] Token IDs and vocabulary
- [ ] Embeddings — learned lookup tables, shape `(vocab_size, D)`
- [ ] Positional Encoding (sinusoidal, learned, RoPE, ALiBi)
- [ ] Why attention is permutation invariant
- [ ] Tokenization quirks (byte-level, whitespace, numbers)

---

## Attention Mechanism

- [ ] Q, K, V — what they represent (query = "what I'm looking for", key = "what I contain", value = "what I offer")
- [ ] Attention Scores: `Q · Kᵀ`
- [ ] Why scale by `1/√d_k` (variance preservation — dot product of d_k-dim vectors has variance d_k)
- [ ] Softmax in Attention (numerical stability)
- [ ] Attention Weights as a soft differentiable lookup
- [ ] Value Aggregation: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
- [ ] Multi-Head Attention — why multiple heads? (different heads learn different relations: syntactic, semantic, positional)
- [ ] Head dimension choice (typically `D_model / H`)
- [ ] Causal masking — upper triangular `-inf` mask
- [ ] Self vs Cross Attention (when to use each)
- [ ] Complexity: `O(N² · D)` time, `O(N²)` space for attention matrix

---

## Core Transformer Architecture

- [ ] Residual Connections — gradient highway, enables deep stacking
- [ ] Layer Normalization (LayerNorm vs RMSNorm)
- [ ] Pre-norm vs Post-norm (modern LLMs use pre-norm for training stability)
- [ ] Feed-Forward Network (MLP/FFN) — typically 2-4x expansion
- [ ] SwiGLU / GeGLU activation (used in LLaMA, PaLM — gated linear units)
- [ ] Full Transformer Block Assembly
- [ ] Masked Self-Attention (causal)
- [ ] Parallel vs Sequential attention+FFN (GPT-J style parallel)
- [ ] Number of layers vs model dimension tradeoffs
- [ ] Tied embeddings (input/output embedding sharing)

---

## Training

- [ ] Next Token Prediction objective
- [ ] Cross Entropy Loss (and why not MSE)
- [ ] Cross Entropy from Logits (log-sum-exp trick)
- [ ] The gradient `p - y` — why it's elegant
- [ ] Backpropagation through Transformers
- [ ] Training vs Inference differences (prefill vs decode, batch vs autoregressive)
- [ ] Mixed precision training (FP16, BF16, loss scaling)
- [ ] Gradient clipping
- [ ] Learning rate scheduling (cosine, warmup)
- [ ] Weight initialization (Xavier/He, small init for residual scaling)
- [ ] Dropout in transformers (attention dropout, residual dropout)

---

## Positional Encoding Deep Dive

- [ ] Sinusoidal — fixed, extrapolates poorly
- [ ] Learned absolute — flexible but no extrapolation
- [ ] RoPE — rotary, relative position via rotation, used in LLaMA/Gemma/Mistral
- [ ] ALiBi — linear bias on attention scores, strong extrapolation
- [ ] RoPE extrapolation methods (NTK-aware, YaRN, position interpolation)
- [ ] Why relative position > absolute position for LLMs

---

## Efficient Attention

- [ ] O(N²) Attention Complexity — the fundamental bottleneck
- [ ] FlashAttention — IO-aware, tiling, online softmax, O(N) space
- [ ] FlashAttention-2 improvements (better parallelism, less overhead)
- [ ] Sliding Window Attention (Mistral uses 4096 window)
- [ ] Sparse Attention patterns (Longformer, BigBird)
- [ ] Linear Attention (approximations — Performer, Linformer)
- [ ] Ring Attention (distribute context across GPUs)

---

## KV Cache & Inference

- [ ] KV Cache — motivation, what's cached, what's not
- [ ] KV Cache memory math: `2 * H_kv * N * D_head * layers * batch * bytes_per_param`
- [ ] PagedAttention (vLLM) — OS-style virtual memory for KV cache
- [ ] Continuous batching — dynamic batch formation during generation
- [ ] Speculative decoding — draft model + verification
- [ ] KV cache quantization (FP8, INT8)
- [ ] Cache eviction strategies (LRU, sliding window)
- [ ] Prefix caching (share KV cache across requests with same system prompt)

---

## Decoding Strategies

- [ ] Greedy Decoding
- [ ] Temperature Scaling
- [ ] Top-K Sampling
- [ ] Top-P (Nucleus) Sampling
- [ ] Beam Search
- [ ] Repetition penalty / frequency penalty / presence penalty
- [ ] Contrastive search
- [ ] Constrained decoding (grammar-guided, JSON mode)
- [ ] Speculative decoding (draft + verify)

---

## Scaling & Architecture

- [ ] Context Window — what limits it, how to extend
- [ ] Scaling Laws (Kaplan vs Chinchilla)
- [ ] Chinchilla rule: ~20 tokens per parameter
- [ ] Overtraining small models for inference efficiency
- [ ] Emergent abilities — real or measurement artifact?
- [ ] Mixture of Experts (MoE) — sparse activation, conditional computation
- [ ] MoE load balancing, expert capacity, routing strategies
- [ ] DeepSeek-style MoE innovations (shared experts, fine-grained experts)

---

## Distributed Training (L5 Critical)

- [ ] Data Parallelism (DP) — replicate model, split data
- [ ] Tensor Parallelism (TP) — split weight matrices across GPUs (Megatron-LM)
- [ ] Pipeline Parallelism (PP) — split layers across GPUs (GPipe, 1F1B)
- [ ] FSDP (Fully Sharded Data Parallel) — shard params/grads/optimizer state
- [ ] ZeRO stages (1: optimizer, 2: +gradients, 3: +params)
- [ ] 3D Parallelism (DP + TP + PP)
- [ ] Sequence Parallelism
- [ ] Gradient synchronization (all-reduce, ring all-reduce)
- [ ] Communication vs computation overlap
- [ ] Mixed precision + distributed training interactions
- [ ] Checkpoint saving in distributed settings

---

## LLM Systems Topics

- [ ] RAG — architecture, chunking, reranking, evaluation
- [ ] Embeddings for Retrieval (dense, sparse, hybrid)
- [ ] Vector Databases (FAISS, HNSW, ScaNN)
- [ ] Fine-Tuning overview (SFT, RLHF, DPO, GRPO)
- [ ] LoRA — motivation + implementation + rank selection
- [ ] QLoRA — quantized base + LoRA adapters
- [ ] Quantization — FP32 / FP16 / BF16 / INT8 / INT4
- [ ] Quantization methods (GPTQ, AWQ, GGUF, SmoothQuant)
- [ ] Hallucinations — causes + mitigations
- [ ] Prompt Engineering patterns (CoT, few-shot, ReAct)
- [ ] Agents vs Workflows (tool use, planning, reflection)
- [ ] Function calling / tool use
- [ ] Context window management (RAG vs long context vs compression)
- [ ] Guardrails and safety filtering
- [ ] Red-teaming and adversarial prompting

---

## Alignment & Safety (L5 Critical)

- [ ] SFT (Supervised Fine-Tuning) — format, style, instruction following
- [ ] RLHF — reward model + PPO, KL penalty
- [ ] DPO — direct preference optimization, no reward model
- [ ] GRPO — group-relative policy optimization (DeepSeek R1)
- [ ] KTO / IPO / SimPO — DPO variants
- [ ] Constitutional AI / RLAIF
- [ ] Reward hacking — causes and mitigations
- [ ] Safety alignment — RLHF for harmlessness
- [ ] Red-teaming methodology
- [ ] Jailbreaks and prompt injection defenses

---

## Serving Infrastructure (L5 Critical)

- [ ] vLLM — PagedAttention, continuous batching
- [ ] TensorRT-LLM — kernel fusion, in-flight batching
- [ ] Model parallelism for serving (TP across GPUs)
- [ ] Batching strategies (static vs dynamic vs continuous)
- [ ] Latency vs throughput tradeoffs
- [ ] TTFT (Time To First Token) vs TPOT (Time Per Output Token)
- [ ] Prefix caching and prompt sharing
- [ ] Multi-tenant serving (priority queues, fair scheduling)
- [ ] Model routing (small model for easy queries, large for hard)
- [ ] Streaming responses
- [ ] Load balancing across GPU workers

---

## Coding Practice

- [ ] Implement Softmax (numerically stable — max subtraction)
- [ ] Implement Cross Entropy from logits (log-sum-exp)
- [ ] Implement Scaled Dot-Product Attention from scratch
- [ ] Implement Multi-Head Attention
- [ ] Implement Causal Masking
- [ ] Implement Embedding Lookup
- [ ] Implement Simple Transformer Block (pre-norm, SwiGLU)
- [ ] Implement RoPE
- [ ] Implement KV Cache (append + attention)
- [ ] Implement Top-K / Top-P sampling
- [ ] Implement LoRA forward pass
- [ ] Implement BPE (train + encode)
- [ ] Implement LayerNorm and RMSNorm
- [ ] Implement GQA/MQA attention
- [ ] Implement MoE routing (Top-K)
- [ ] Complexity Analysis Questions (derive O() for each)

---

## L5 System Design Questions

For each question, be ready to: (1) clarify requirements, (2) propose architecture, (3) discuss tradeoffs, (4) dive deep on any component, (5) discuss failure modes and mitigations.

### Serving & Deployment

- **How would you serve a 70B model with limited GPU memory?**
  - Quantization (INT4/INT8), tensor parallelism across multiple GPUs, LoRA for task-specific adapters, KV cache management
- **Design an LLM serving system that handles 1000 concurrent requests.**
  - Continuous batching, PagedAttention, prefix caching, priority queues, model routing, autoscaling
- **How would you reduce TTFT for a chatbot serving a 13B model?**
  - Prefill optimization, prefix caching for system prompts, speculative decoding, smaller draft model
- **How do you handle variable-length sequences in a batched serving system?**
  - Dynamic padding, continuous batching (vLLM), PagedAttention for memory efficiency

### Training & Fine-Tuning

- **How would you fine-tune a 70B model on a single 8-GPU node?**
  - QLoRA (INT4 base + LoRA), gradient checkpointing, FSDP, offloading
- **Design a distributed training system for a 175B model across 1024 GPUs.**
  - 3D parallelism (TP=8, PP=16, DP=8), ZeRO optimizer, gradient checkpointing, communication overlap
- **When would you use LoRA vs full fine-tuning vs continued pretraining?**
  - LoRA: task adaptation, limited compute. Full FT: domain shift, large budget. Continued PT: new language/domain.

### RAG & Applications

- **Design a RAG system for a company's internal knowledge base with 10M documents.**
  - Chunking strategy, embedding model, ANN index (HNSW), reranking, hybrid search, evaluation pipeline, incremental indexing
- **How would you evaluate retrieval quality in a RAG system?**
  - Recall@K, MRR, end-to-end faithfulness, RAGAS framework, human eval for final answer quality
- **How would you handle multi-turn conversations in a RAG system?**
  - Query rewriting, conversation memory, hybrid retrieval (current turn + context)

### Alignment & Safety

- **What causes hallucinations and how do you reduce them?**
  - Training data noise, knowledge gaps, sampling randomness. Mitigations: RAG, lower temperature, grounding prompts, fact-checking, constrained decoding
- **Design an alignment pipeline for a customer-facing chatbot.**
  - SFT → DPO/RLHF → red-teaming → safety classifier → monitoring → feedback loop
- **How would you detect and prevent reward hacking in RLHF?**
  - KL penalty, reward model ensembles, adversarial testing, monitoring reward vs quality divergence

### Architecture & Scaling

- **Why does GPT-3 underperform compared to smaller newer models?**
  - Undertrained (300B tokens for 175B params vs Chinchilla optimal ~3.5T)
- **When would you choose MoE over a dense model?**
  - When you want more capacity without proportional compute increase. Tradeoff: memory, routing complexity, load balancing
- **How would you extend a model's context window from 4K to 128K?**
  - RoPE scaling (NTK-aware, YaRN), FlashAttention for memory, sliding window + global attention, RAG as alternative

---

## Interview Strategy for L5

1. **Start with the big picture** — always frame your answer in terms of the system, not just the algorithm
2. **Discuss tradeoffs explicitly** — "The tradeoff here is X vs Y, and I'd choose X because..."
3. **Know your numbers** — memory math (7B in FP16 = 14GB), KV cache sizes, Chinchilla ratio, attention complexity
4. **Be ready to implement** — if asked to code, write clean, correct, numerically stable code
5. **Connect topics** — show how FlashAttention enables long context, how GQA reduces KV cache which helps serving, how quantization + LoRA enable edge deployment
6. **Admit uncertainty gracefully** — "I'm not 100% sure, but my intuition is..." is better than bluffing
7. **Ask clarifying questions** — L5 is about system thinking, not just recall
