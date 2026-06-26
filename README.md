# ML & LLM Interview Prep

A structured preparation repository for **Google L5 ML/LLM Engineering** interviews.
Covers core ML algorithms, LLM/Transformer internals, and ML system design.

Every code file is self-contained, heavily commented, and runnable with minimal dependencies (`numpy` for ML, `torch` for LLM/Transformer implementations).

---

## Repository Structure

```
algo/
├── code/
│   ├── ml/                          # Core ML algorithms (NumPy only)
│   │   ├── linear_regression.py     #   Multi-feature linear regression (MSE + GD)
│   │   ├── logistic_regression.py   #   Logistic regression (BCE + GD)
│   │   ├── knn.py                   #   K-Nearest Neighbors (lazy learner)
│   │   ├── kmeans.py                #   K-Means clustering (Lloyd's algorithm)
│   │   ├── softmax.py               #   Softmax regression (multiclass CE + GD)
│   │   ├── cross_entropy.py         #   Cross-entropy loss demo (correct vs wrong)
│   │   ├── neural_network.py        #   2-layer MLP forward pass (ReLU + Softmax)
│   │   ├── backpropagation.py       #   2-layer MLP with full backprop training
│   │   ├── relu.py                  #   ReLU vs Sigmoid (vanishing gradient, dead ReLU)
│   │   ├── metrics.py               #   Accuracy, precision, recall, F1, ROC-AUC
│   │   ├── bias_variance.py         #   Underfit / good-fit / overfit demo
│   │   ├── regularization.py        #   L1/L2 regularization (Ridge closed-form)
│   │   └── data_splits.py           #   Train/val/test split + data leakage
│   │
│   ├── llm/
│   │   ├── numpy/                   # LLM components in pure NumPy
│   │   │   ├── softmax.py           #   Numerically stable softmax
│   │   │   ├── attention.py         #   Scaled dot-product attention + causal mask
│   │   │   ├── multihead_attention.py #  Multi-Head Attention (MHA)
│   │   │   ├── mqa.py               #   Multi-Query Attention (MQA)
│   │   │   ├── gqa.py               #   Grouped-Query Attention (GQA)
│   │   │   ├── layer_norm.py        #   LayerNorm + RMSNorm
│   │   │   ├── cross_entropy.py     #   CE from probabilities
│   │   │   ├── cross_entropy_logits.py # CE from logits (log-sum-exp trick)
│   │   │   ├── perplexity.py        #   Perplexity = exp(CE)
│   │   │   ├── cosine_similarity.py #   Cosine similarity + retrieval
│   │   │   ├── sampling.py          #   Temperature, top-k, top-p decoding
│   │   │   ├── kv_cache.py          #   KV cache for autoregressive decoding
│   │   │   ├── lora.py              #   LoRA low-rank adaptation
│   │   │   ├── moe.py               #   Mixture of Experts forward pass
│   │   │   ├── rope.py              #   Rotary Positional Embeddings
│   │   │   └── transformer.py       #   Full GPT-style transformer (educational)
│   │   │
│   │   └── torch/                   # LLM components in PyTorch
│   │       ├── transformer_block.py #  Transformer block (MHA + FFN + residuals)
│   │       ├── gpt_model.py         #  GPT model with weight tying
│   │       ├── gqa.py               #  GQA as nn.Module
│   │       ├── mqa.py               #  MQA as nn.Module
│   │       ├── rope.py              #  RoPE (functional implementation)
│   │       ├── rope_module.py       #  RoPE as nn.Module (registered buffers)
│   │       ├── rlhf.py              #  PPO, DPO, GRPO loss functions
│   │       ├── kv_cache.py          #  KV cache as nn.Module
│   │       └── lora.py              #  LoRALinear nn.Module + weight merging
│   │
│   └── practice/                    # Coding practice (not interview-critical)
│       ├── practice.py
│       ├── trials.py
│       └── test.py
│
├── notes/
│   ├── ml/                          # ML algorithm deep-dive notes
│   │   ├── Volume 1 - Linear Regression & Gradient Descent.md
│   │   ├── Volume 2 - Logistic Regression.md
│   │   ├── Volume 3 - K-Means.md
│   │   ├── Volume 4 - KNN.md
│   │   ├── Linear Regression.md
│   │   ├── Logistic Regression.md
│   │   ├── K-Means.md
│   │   ├── KNN.md
│   │   └── Universal ML Interview Notes.md
│   │
│   ├── llm/                         # LLM/Transformer interview notes (18 files)
│   │   ├── 00_LLM Transformer Interview Roadmap.md
│   │   ├── 01_Tokenization & BPE.md
│   │   ├── 02_Cross Entropy.md
│   │   ├── 03_Cross Entropy From Logits.md
│   │   ├── 04_Transformer Advanced Topics.md
│   │   ├── 05_RoPE Implementation.md
│   │   ├── 06_GQA & MQA.md
│   │   ├── 07_FlashAttention Intuition.md
│   │   ├── 08_Perplexity, LayerNorm & RMSNorm.md
│   │   ├── 09_KV Cache & Decoding.md
│   │   ├── 10_KV Cache Implementation.md
│   │   ├── 11_RAG.md
│   │   ├── 12_LoRA & Quantization.md
│   │   ├── 13_LoRA Forward Pass.md
│   │   ├── 14_RLHF & DPO.md
│   │   ├── 15_Scaling Laws.md
│   │   ├── 16_LLM Evaluation Metrics.md
│   │   └── 17_MoE (Mixture of Experts).md
│   │
│   ├── system_design/               # ML system design foundations
│   │   ├── ML System Design Mental Framework.md
│   │   ├── System Design Foundations - Session 1.md   # Requirements
│   │   ├── System Design Foundations - Session 2.md   # Tradeoffs & cost of error
│   │   ├── System Design Foundations - Session 3.md   # Scale estimation
│   │   ├── System Design Foundations - Session 4.md   # Distributed systems
│   │   └── System Design Foundations - Session 5.md   # Infrastructure components
│   │
│   ├── cheat_sheets/                # Interview pattern cheat sheets
│   │   ├── Google ML Coding Interview Cheat Sheet (LLM Focused).md
│   │   ├── Google ML Coding Interview Pattern Cheat Sheet.md
│   │   └── The Pattern Detector.md
│   │
│   └── reference/                   # Study guides & schedules
│       ├── DATA_DESIGN.md           #   Dataset design philosophy
│       ├── ML_Interview_Prep_Tracker.md  # Daily ritual tracker
│       └── SCHEDULE.md              #   12-day prep schedule
│
└── .venv/                           # Python virtual environment
```

---

## Code Organization Principles

1. **One topic per file** — no mixed concerns. If a concept needs its own file, it gets one.
2. **NumPy vs PyTorch split** — `code/llm/numpy/` for educational implementations (no framework magic), `code/llm/torch/` for production-style `nn.Module` code.
3. **Heavily commented** — every file has a module docstring, function docstrings, shape annotations, and inline `# why` comments explaining the reasoning, not just the mechanics.
4. **Runnable** — every file can be executed standalone (`python file.py`) with minimal dependencies.
5. **No new functionality** — code logic matches original implementations; comments and structure are the improvements.

---

## Prerequisites

```bash
pip install numpy torch scikit-learn
```

- `code/ml/` — requires only `numpy` (and `scikit-learn` for ROC-AUC in `metrics.py`)
- `code/llm/numpy/` — requires only `numpy`
- `code/llm/torch/` — requires `torch`

---

## How to Use This Repo

### If you're preparing for an ML/LLM interview

1. **Read the notes first** — start with `notes/llm/00_LLM Transformer Interview Roadmap.md` for the full topic list, then work through `01`–`17` in order.
2. **Implement from memory** — after reading a note, try to write the corresponding code file from scratch. Compare with the implementation in `code/`.
3. **Oral drill** — use `notes/reference/ML_Interview_Prep_Tracker.md` for daily oral drill checklists. Explain each concept out loud as if in an interview.
4. **System design** — read `notes/system_design/` sessions 1–5 in order, then practice with the mental framework.
5. **Cheat sheets** — before the interview, skim `notes/cheat_sheets/` for pattern recognition.

### If you're learning ML/LLM fundamentals

1. Start with `code/ml/` — these are the building blocks. Each file is self-contained and demonstrates one concept with a tiny, deterministic dataset.
2. Move to `code/llm/numpy/` — see how Transformer components work without framework abstractions.
3. Then `code/llm/torch/` — see the same concepts as `nn.Module` classes, closer to real-world usage.
4. Read the corresponding `notes/` file for deeper theory, derivations, and interview Q&A.

### If you're reviewing before an interview

- **Day 1–2**: Re-implement all `code/ml/` files from memory.
- **Day 3–5**: Re-implement `code/llm/numpy/` files from memory.
- **Day 6–7**: Walk through `code/llm/torch/` files, explain each line.
- **Day 8–10**: Read all `notes/llm/` files, do oral drills.
- **Day 11–12**: Practice system design using `notes/system_design/` framework.

---

## Topic Coverage

### ML Algorithms (`code/ml/`)

| File | Topic | Key Concepts |
|------|-------|--------------|
| `linear_regression.py` | Linear regression | MSE loss, gradient descent, closed-form vs iterative |
| `logistic_regression.py` | Logistic regression | Sigmoid, BCE loss, (p − y) gradient |
| `knn.py` | K-Nearest Neighbors | Euclidean distance, majority vote, lazy learning |
| `kmeans.py` | K-Means clustering | Lloyd's algorithm, assignment + update, convergence |
| `softmax.py` | Softmax regression | Multiclass CE, (p − y_onehot) gradient, stable softmax |
| `cross_entropy.py` | Cross-entropy loss | CE formula, confidently-wrong penalty, log(0) prevention |
| `neural_network.py` | Neural network forward pass | MLP, ReLU, softmax, shape flow |
| `backpropagation.py` | Backpropagation | Chain rule, gradient flow, ReLU gradient |
| `relu.py` | ReLU activation | Vanishing gradient, dead ReLU, ReLU vs sigmoid |
| `metrics.py` | Classification metrics | Confusion matrix, precision/recall/F1, ROC-AUC |
| `bias_variance.py` | Bias-variance tradeoff | Underfitting, good fit, overfitting |
| `regularization.py` | L1/L2 regularization | Ridge closed-form, Lasso sparsity, weight explosion |
| `data_splits.py` | Train/val/test splits | Data leakage, split ratios, reproducibility |

### LLM Components — NumPy (`code/llm/numpy/`)

| File | Topic | Key Concepts |
|------|-------|--------------|
| `softmax.py` | Softmax | Numerical stability, max-shift trick |
| `attention.py` | Scaled dot-product attention | Causal masking, sqrt(d_k) scaling |
| `multihead_attention.py` | Multi-Head Attention | Head splitting, reshape/transpose, output projection |
| `mqa.py` | Multi-Query Attention | Single KV head, KV cache reduction |
| `gqa.py` | Grouped-Query Attention | KV head groups, memory vs quality tradeoff |
| `layer_norm.py` | LayerNorm & RMSNorm | Per-token normalization, why not BatchNorm |
| `cross_entropy.py` | Cross-entropy from probs | Sparse labels, clipping, CE formula |
| `cross_entropy_logits.py` | CE from logits | Log-sum-exp trick, numerical stability |
| `perplexity.py` | Perplexity | PPL = exp(CE), interpretation |
| `cosine_similarity.py` | Cosine similarity & retrieval | Dot product of normalized vectors, top-k retrieval |
| `sampling.py` | Decoding strategies | Temperature, top-k, top-p (nucleus) sampling |
| `kv_cache.py` | KV cache | O(N²) → O(N) decoding, cache append |
| `lora.py` | LoRA | Low-rank adaptation, parameter reduction, A/B matrices |
| `moe.py` | Mixture of Experts | Router, top-k selection, weighted expert combination |
| `rope.py` | Rotary Positional Embeddings | Rotation matrix, inv_freq, relative position encoding |
| `transformer.py` | Full transformer | Embedding → blocks → vocab projection, architecture overview |

### LLM Components — PyTorch (`code/llm/torch/`)

| File | Topic | Key Concepts |
|------|-------|--------------|
| `transformer_block.py` | Transformer block | MHA + FFN + residuals + LayerNorm as `nn.Module` |
| `gpt_model.py` | GPT model | Token/positional embedding, weight tying, stacked blocks |
| `gqa.py` | GQA module | `repeat_interleave`, causal mask, KV head grouping |
| `mqa.py` | MQA module | Single KV head expansion via `expand` |
| `rope.py` | RoPE (functional) | `torch.outer`, precomputed cos/sin, apply to Q and K |
| `rope_module.py` | RoPE (nn.Module) | Registered buffers, device-aware, state_dict serialization |
| `rlhf.py` | RLHF losses | PPO clip ratio, DPO Bradley-Terry, GRPO group normalization |
| `kv_cache.py` | KV cache module | `torch.cat` append, reset, `nn.Module` wrapper |
| `lora.py` | LoRA module | `LoRALinear` class, zero-init B, weight merging for inference |

### Notes — LLM (`notes/llm/`)

| # | Topic | L5 Interview Q&A |
|---|-------|------------------|
| 00 | Roadmap | Full topic checklist |
| 01 | Tokenization & BPE | WordPiece, SentencePiece, vocab size tradeoffs |
| 02 | Cross Entropy | KL divergence, label smoothing, focal loss |
| 03 | CE From Logits | Fused implementation, gradient computation, vocab-level CE |
| 04 | Transformer Advanced | ALiBi, pre/post-norm, SwiGLU, sliding window, FlashAttention-2 |
| 05 | RoPE Implementation | Math derivation, interleaved vs half-split, scaling methods |
| 06 | GQA & MQA | KV cache math, quality benchmarks, when to use |
| 07 | FlashAttention | Online softmax, tiling, FA-2/3 improvements |
| 08 | Perplexity, LayerNorm & RMSNorm | Calibration, norm comparisons, gradient analysis |
| 09 | KV Cache & Decoding | PagedAttention, speculative decoding, penalties |
| 10 | KV Cache Implementation | Pre-allocation, multi-layer, GQA-aware, prefix caching |
| 11 | RAG | Hybrid search, reranking, RAGAS, multi-turn, chunking |
| 12 | LoRA & Quantization | QLoRA, rank selection, BF16 vs FP16, quantization methods |
| 13 | LoRA Forward Pass | Initialization, gradient flow, multi-adapter serving |
| 14 | RLHF & DPO | GRPO, KTO/IPO/SimPO, reward hacking, RLAIF |
| 15 | Scaling Laws | Chinchilla, overtraining economics, compute-optimal |
| 16 | LLM Evaluation | G-Eval, MT-Bench, AGIEval, data contamination, LLM-as-judge |
| 17 | MoE | DeepSeek MoE, load balancing, capacity factor, MoE serving |

### Notes — System Design (`notes/system_design/`)

| Session | Topic | L5 Interview Q&A |
|---------|-------|------------------|
| Framework | ML System Design Mental Framework | 11-step framework, LLM patterns |
| 1 | Requirements | Functional/NFR, removal test, ML-specific requirements |
| 2 | Tradeoffs | Quality/latency/cost/freshness, cost of error, metrics hierarchy |
| 3 | Scale Estimation | GPU memory, LLM serving capacity, vector DB storage, training compute |
| 4 | Distributed Systems | Training consistency, model replication, KV cache, rollouts, failure modes |
| 5 | Infrastructure | GPU clusters, vector DBs, feature stores, model registry, caching, monitoring |

---

## Study Schedule

A 12-day intensive plan is in `notes/reference/SCHEDULE.md`. The daily ritual:

1. **Re-implement** 2–3 core algorithms from memory (no looking at code).
2. **Oral drill** — explain the algorithm out loud as if teaching someone.
3. **Read** the corresponding note file for deeper theory and L5 Q&A.
4. **Cumulative review** — re-explain all previously covered topics.

---

## Dataset Design Philosophy

All code files use tiny, deterministic datasets designed for interview settings:
- **Visually obvious** — you can eyeball the expected answer.
- **Deterministic** — no random data (or fixed seed) so output is reproducible.
- **Small enough to debug by hand** — typically 4–9 data points.
- **Easy to explain** — in an interview, you can write the dataset on a whiteboard.

See `notes/reference/DATA_DESIGN.md` for the full philosophy.

---

## Running the Code

```bash
# ML algorithms (numpy only)
python code/ml/linear_regression.py
python code/ml/backpropagation.py

# LLM NumPy implementations
python code/llm/numpy/attention.py
python code/llm/numpy/transformer.py

# LLM PyTorch implementations
python code/llm/torch/gqa.py
python code/llm/torch/rlhf.py
```

Each file prints its output and expected results.

---

## License

Personal study material. Not affiliated with Google.
