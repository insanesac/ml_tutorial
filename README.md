# ML & LLM Interview Prep

A structured preparation repository for **Google L5 ML/LLM Engineering** interviews.
Covers core ML algorithms, LLM/Transformer internals, and ML system design.

Every code file is self-contained, heavily commented, and runnable with minimal dependencies (`numpy` for ML, `torch` for LLM/Transformer implementations).

---

## Repository Structure

```
algo/
├── AI_ENGINEER_ROADMAP.md          # Master roadmap (15 modules, learning phases)
├── code/
│   ├── 00_ml/                       # Core ML algorithms (NumPy only)
│   │   ├── 00_linear_regression.py  #   Multi-feature linear regression (MSE + GD)
│   │   ├── 01_logistic_regression.py#   Logistic regression (BCE + GD)
│   │   ├── 02_knn.py                #   K-Nearest Neighbors (lazy learner)
│   │   ├── 03_kmeans.py             #   K-Means clustering (Lloyd's algorithm)
│   │   ├── 04_softmax.py            #   Softmax regression (multiclass CE + GD)
│   │   ├── 05_cross_entropy.py      #   Cross-entropy loss demo (correct vs wrong)
│   │   ├── 06_neural_network.py     #   2-layer MLP forward pass (ReLU + Softmax)
│   │   ├── 07_relu.py               #   ReLU vs Sigmoid (vanishing gradient, dead ReLU)
│   │   ├── 08_backpropagation.py    #   2-layer MLP with full backprop training
│   │   ├── 09_metrics.py            #   Accuracy, precision, recall, F1, ROC-AUC
│   │   ├── 10_bias_variance.py      #   Underfit / good-fit / overfit demo
│   │   ├── 11_regularization.py     #   L1/L2 regularization (Ridge closed-form)
│   │   └── 12_data_splits.py        #   Train/val/test split + data leakage
│   │
│   ├── 01_llm_numpy/                # LLM components in pure NumPy
│   │   ├── 00_softmax.py            #   Numerically stable softmax
│   │   ├── 01_attention.py          #   Scaled dot-product attention + causal mask
│   │   ├── 02_multihead_attention.py#  Multi-Head Attention (MHA)
│   │   ├── 03_mqa.py                #   Multi-Query Attention (MQA)
│   │   ├── 04_gqa.py                #   Grouped-Query Attention (GQA)
│   │   ├── 05_rope.py               #   Rotary Positional Embeddings
│   │   ├── 06_layer_norm.py         #   LayerNorm + RMSNorm
│   │   ├── 07_cross_entropy.py      #   CE from probabilities
│   │   ├── 08_cross_entropy_logits.py#  CE from logits (log-sum-exp trick)
│   │   ├── 09_perplexity.py         #   Perplexity = exp(CE)
│   │   ├── 10_cosine_similarity.py  #   Cosine similarity + retrieval
│   │   ├── 11_sampling.py           #   Temperature, top-k, top-p decoding
│   │   ├── 12_kv_cache.py           #   KV cache for autoregressive decoding
│   │   ├── 13_lora.py               #   LoRA low-rank adaptation
│   │   ├── 14_moe.py                #   Mixture of Experts forward pass
│   │   └── 15_transformer.py        #   Full GPT-style transformer (educational)
│   │
│   └── 02_llm_torch/                # LLM components in PyTorch
│       ├── 00_transformer_block.py  #  Transformer block (MHA + FFN + residuals)
│       ├── 01_gpt_model.py          #  GPT model with weight tying
│       ├── 02_gqa.py                #  GQA as nn.Module
│       ├── 03_mqa.py                #  MQA as nn.Module
│       ├── 04_rope.py               #  RoPE (functional implementation)
│       ├── 05_rope_module.py        #  RoPE as nn.Module (registered buffers)
│       ├── 06_kv_cache.py           #  KV cache as nn.Module
│       ├── 07_lora.py               #  LoRALinear nn.Module + weight merging
│       └── 08_rlhf.py               #  PPO, DPO, GRPO loss functions
│
├── notes/
│   ├── 00_ml/                      # ML algorithm deep-dive notes
│   │   ├── 00_universal_ml_notes.md #   Shape rules, complexity, oral drill checklist
│   │   ├── 01_linear_regression.md  #   Linear regression & gradient descent
│   │   ├── 02_logistic_regression.md#   Logistic regression, sigmoid, BCE
│   │   ├── 03_kmeans.md             #   K-Means clustering, WCSS, elbow method
│   │   └── 04_knn.md               #   KNN, lazy learning, bias-variance via K
│   │
│   ├── 01_llm/                      # LLM/Transformer interview notes (19 files)
│   │   ├── 00_LLM Transformer Interview Roadmap.md
│   │   ├── 01_Tokenization & BPE.md
│   │   ├── 02_Cross Entropy.md
│   │   ├── 03_Cross Entropy From Logits.md
│   │   ├── 04_Backpropagation.md     #   Chain rule, autograd, activation storage
│   │   ├── 05_Transformer Advanced Topics.md
│   │   ├── 06_RoPE Implementation.md
│   │   ├── 07_GQA & MQA.md
│   │   ├── 08_FlashAttention Intuition.md
│   │   ├── 09_Perplexity, LayerNorm & RMSNorm.md
│   │   ├── 10_KV Cache & Decoding.md
│   │   ├── 11_KV Cache Implementation.md
│   │   ├── 12_RAG.md
│   │   ├── 13_LoRA & Quantization.md
│   │   ├── 14_LoRA Forward Pass.md
│   │   ├── 15_RLHF & DPO.md
│   │   ├── 16_Scaling Laws.md
│   │   ├── 17_LLM Evaluation Metrics.md
│   │   └── 18_MoE (Mixture of Experts).md
│   │
│   ├── 02_system_design/            # ML system design foundations
│   │   ├── 00_system_design_framework.md  # 11-step mental framework
│   │   ├── 01_requirements.md       #   Functional/NFR, ML-specific requirements
│   │   ├── 02_tradeoffs.md          #   Quality/latency/cost, cost of error
│   │   ├── 03_scale_estimation.md   #   GPU memory, serving capacity, storage
│   │   ├── 04_distributed_systems.md#   Training consistency, replication, failure
│   │   └── 05_infrastructure.md     #   GPU clusters, vector DBs, monitoring
│   │
│   ├── 03_cheat_sheets/             # Interview pattern cheat sheets
│   │   ├── 00_llm_coding_cheatsheet.md  # LLM-focused coding cheat sheet
│   │   ├── 01_pattern_cheatsheet.md     # ML coding interview patterns
│   │   └── 02_pattern_detector.md       # Pattern recognition for interviews
│   │
│   └── 04_reference/                # Study guides & schedules
│       ├── 00_schedule.md           #   12-day intensive prep schedule
│       ├── 01_prep_tracker.md       #   Daily ritual tracker & oral drill checklist
│       └── 02_data_design.md        #   Dataset design philosophy
│
└── .venv/                           # Python virtual environment
```

---

## Code Organization Principles

1. **One topic per file** — no mixed concerns. If a concept needs its own file, it gets one.
2. **NumPy vs PyTorch split** — `code/01_llm_numpy/` for educational implementations (no framework magic), `code/02_llm_torch/` for production-style `nn.Module` code.
3. **Heavily commented** — every file has a module docstring, function docstrings, shape annotations, and inline `# why` comments explaining the reasoning, not just the mechanics.
4. **Runnable** — every file can be executed standalone (`python file.py`) with minimal dependencies. Files are self-contained (no cross-file imports).
5. **No new functionality** — code logic matches original implementations; comments and structure are the improvements.

---

## Prerequisites

```bash
pip install numpy torch scikit-learn
```

- `code/00_ml/` — requires only `numpy` (and `scikit-learn` for ROC-AUC in `09_metrics.py`)
- `code/01_llm_numpy/` — requires only `numpy`
- `code/02_llm_torch/` — requires `torch`

---

## How to Use This Repo

### If you're preparing for an ML/LLM interview

1. **Read the notes first** — start with `AI_ENGINEER_ROADMAP.md` for the full 15-module roadmap, then `notes/01_llm/00_LLM Transformer Interview Roadmap.md` for LLM-specific topics, then work through `01`–`18` in order.
2. **Implement from memory** — after reading a note, try to write the corresponding code file from scratch. Compare with the implementation in `code/`.
3. **Oral drill** — use `notes/04_reference/01_prep_tracker.md` for daily oral drill checklists. Explain each concept out loud as if in an interview.
4. **System design** — read `notes/02_system_design/` sessions 0–5 in order, then practice with the mental framework.
5. **Cheat sheets** — before the interview, skim `notes/03_cheat_sheets/` for pattern recognition.

### If you're learning ML/LLM fundamentals

1. Start with `code/00_ml/` — these are the building blocks. Each file is self-contained and demonstrates one concept with a tiny, deterministic dataset.
2. Move to `code/01_llm_numpy/` — see how Transformer components work without framework abstractions.
3. Then `code/02_llm_torch/` — see the same concepts as `nn.Module` classes, closer to real-world usage.
4. Read the corresponding `notes/` file for deeper theory, derivations, and interview Q&A.

### If you're reviewing before an interview

- **Day 1–2**: Re-implement all `code/00_ml/` files from memory.
- **Day 3–5**: Re-implement `code/01_llm_numpy/` files from memory.
- **Day 6–7**: Walk through `code/02_llm_torch/` files, explain each line.
- **Day 8–10**: Read all `notes/01_llm/` files, do oral drills.
- **Day 11–12**: Practice system design using `notes/02_system_design/` framework.

---

## Topic Coverage

### ML Algorithms (`code/00_ml/`)

| File | Topic | Key Concepts |
|------|-------|--------------|
| `00_linear_regression.py` | Linear regression | MSE loss, gradient descent, closed-form vs iterative |
| `01_logistic_regression.py` | Logistic regression | Sigmoid, BCE loss, (p − y) gradient |
| `02_knn.py` | K-Nearest Neighbors | Euclidean distance, majority vote, lazy learning |
| `03_kmeans.py` | K-Means clustering | Lloyd's algorithm, assignment + update, convergence |
| `04_softmax.py` | Softmax regression | Multiclass CE, (p − y_onehot) gradient, stable softmax |
| `05_cross_entropy.py` | Cross-entropy loss | CE formula, confidently-wrong penalty, log(0) prevention |
| `06_neural_network.py` | Neural network forward pass | MLP, ReLU, softmax, shape flow |
| `07_relu.py` | ReLU activation | Vanishing gradient, dead ReLU, ReLU vs sigmoid |
| `08_backpropagation.py` | Backpropagation | Chain rule, gradient flow, ReLU gradient |
| `09_metrics.py` | Classification metrics | Confusion matrix, precision/recall/F1, ROC-AUC |
| `10_bias_variance.py` | Bias-variance tradeoff | Underfitting, good fit, overfitting |
| `11_regularization.py` | L1/L2 regularization | Ridge closed-form, Lasso sparsity, weight explosion |
| `12_data_splits.py` | Train/val/test splits | Data leakage, split ratios, reproducibility |

### LLM Components — NumPy (`code/01_llm_numpy/`)

| File | Topic | Key Concepts |
|------|-------|--------------|
| `00_softmax.py` | Softmax | Numerical stability, max-shift trick |
| `01_attention.py` | Scaled dot-product attention | Causal masking, sqrt(d_k) scaling |
| `02_multihead_attention.py` | Multi-Head Attention | Head splitting, reshape/transpose, output projection |
| `03_mqa.py` | Multi-Query Attention | Single KV head, KV cache reduction |
| `04_gqa.py` | Grouped-Query Attention | KV head groups, memory vs quality tradeoff |
| `05_rope.py` | Rotary Positional Embeddings | Rotation matrix, inv_freq, relative position encoding |
| `06_layer_norm.py` | LayerNorm & RMSNorm | Per-token normalization, why not BatchNorm |
| `07_cross_entropy.py` | Cross-entropy from probs | Sparse labels, clipping, CE formula |
| `08_cross_entropy_logits.py` | CE from logits | Log-sum-exp trick, numerical stability |
| `09_perplexity.py` | Perplexity | PPL = exp(CE), interpretation |
| `10_cosine_similarity.py` | Cosine similarity & retrieval | Dot product of normalized vectors, top-k retrieval |
| `11_sampling.py` | Decoding strategies | Temperature, top-k, top-p (nucleus) sampling |
| `12_kv_cache.py` | KV cache | O(N²) → O(N) decoding, cache append |
| `13_lora.py` | LoRA | Low-rank adaptation, parameter reduction, A/B matrices |
| `14_moe.py` | Mixture of Experts | Router, top-k selection, weighted expert combination |
| `15_transformer.py` | Full transformer | Embedding → blocks → vocab projection, architecture overview |

### LLM Components — PyTorch (`code/02_llm_torch/`)

| File | Topic | Key Concepts |
|------|-------|--------------|
| `00_transformer_block.py` | Transformer block | MHA + FFN + residuals + LayerNorm as `nn.Module` |
| `01_gpt_model.py` | GPT model | Token/positional embedding, weight tying, stacked blocks |
| `02_gqa.py` | GQA module | `repeat_interleave`, causal mask, KV head grouping |
| `03_mqa.py` | MQA module | Single KV head expansion via `expand` |
| `04_rope.py` | RoPE (functional) | `torch.outer`, precomputed cos/sin, apply to Q and K |
| `05_rope_module.py` | RoPE (nn.Module) | Registered buffers, device-aware, state_dict serialization |
| `06_kv_cache.py` | KV cache module | `torch.cat` append, reset, `nn.Module` wrapper |
| `07_lora.py` | LoRA module | `LoRALinear` class, zero-init B, weight merging for inference |
| `08_rlhf.py` | RLHF losses | PPO clip ratio, DPO Bradley-Terry, GRPO group normalization |

### Notes — ML (`notes/00_ml/`)

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 00 | `00_universal_ml_notes.md` | Universal ML notes | Shape rules, complexity rules, oral drill checklist |
| 01 | `01_linear_regression.md` | Linear regression | MSE, gradient descent, chain rule intuition, learning rate |
| 02 | `02_logistic_regression.md` | Logistic regression | Sigmoid, log-odds, BCE, (p − y) gradient |
| 03 | `03_kmeans.md` | K-Means | WCSS, assignment vs update, elbow method, outlier sensitivity |
| 04 | `04_knn.md` | KNN | Lazy learner, bias-variance via K, complexity |

### Notes — LLM (`notes/01_llm/`)

| # | Topic | L5 Interview Q&A |
|---|-------|------------------|
| 00 | Roadmap | Full topic checklist |
| 01 | Tokenization & BPE | WordPiece, SentencePiece, vocab size tradeoffs |
| 02 | Cross Entropy | KL divergence, label smoothing, focal loss |
| 03 | CE From Logits | Fused implementation, gradient computation, vocab-level CE |
| 04 | Backpropagation | Chain rule, computational graph, activation storage, PyTorch autograd |
| 05 | Transformer Advanced | ALiBi, pre/post-norm, SwiGLU, sliding window, FlashAttention-2 |
| 06 | RoPE Implementation | Math derivation, interleaved vs half-split, scaling methods |
| 07 | GQA & MQA | KV cache math, quality benchmarks, when to use |
| 08 | FlashAttention | Online softmax, tiling, FA-2/3 improvements |
| 09 | Perplexity, LayerNorm & RMSNorm | Calibration, norm comparisons, gradient analysis |
| 10 | KV Cache & Decoding | PagedAttention, speculative decoding, penalties |
| 11 | KV Cache Implementation | Pre-allocation, multi-layer, GQA-aware, prefix caching |
| 12 | RAG | Hybrid search, reranking, RAGAS, multi-turn, chunking |
| 13 | LoRA & Quantization | QLoRA, rank selection, BF16 vs FP16, quantization methods |
| 14 | LoRA Forward Pass | Initialization, gradient flow, multi-adapter serving |
| 15 | RLHF & DPO | GRPO, KTO/IPO/SimPO, reward hacking, RLAIF |
| 16 | Scaling Laws | Chinchilla, overtraining economics, compute-optimal |
| 17 | LLM Evaluation | G-Eval, MT-Bench, AGIEval, data contamination, LLM-as-judge |
| 18 | MoE | DeepSeek MoE, load balancing, capacity factor, MoE serving |

### Notes — System Design (`notes/02_system_design/`)

| # | File | Topic | L5 Interview Q&A |
|---|------|-------|------------------|
| 00 | `00_system_design_framework.md` | Mental framework | 11-step framework, LLM patterns |
| 01 | `01_requirements.md` | Requirements | Functional/NFR, removal test, ML-specific requirements |
| 02 | `02_tradeoffs.md` | Tradeoffs | Quality/latency/cost/freshness, cost of error, metrics hierarchy |
| 03 | `03_scale_estimation.md` | Scale estimation | GPU memory, LLM serving capacity, vector DB storage, training compute |
| 04 | `04_distributed_systems.md` | Distributed systems | Training consistency, model replication, KV cache, rollouts, failure modes |
| 05 | `05_infrastructure.md` | Infrastructure | GPU clusters, vector DBs, feature stores, model registry, caching, monitoring |

---

## Study Schedule

A 12-day intensive plan is in `notes/04_reference/00_schedule.md`. The daily ritual:

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

See `notes/04_reference/02_data_design.md` for the full philosophy.

---

## Running the Code

```bash
# ML algorithms (numpy only)
python code/00_ml/00_linear_regression.py
python code/00_ml/08_backpropagation.py

# LLM NumPy implementations
python code/01_llm_numpy/01_attention.py
python code/01_llm_numpy/15_transformer.py

# LLM PyTorch implementations
python code/02_llm_torch/02_gqa.py
python code/02_llm_torch/08_rlhf.py
```

Each file prints its output and expected results.

---

## License

Personal study material. Not affiliated with Google.
