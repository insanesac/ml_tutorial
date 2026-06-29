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
│       ├── 08_rlhf.py               #  PPO, DPO, GRPO loss functions
│       └── 09_swiglu.py             #  SwiGLU FFN + LLaMA-style transformer block
│
├── notes/
│   ├── 00_math/                    # Math foundations (read first)
│   │   ├── linear_algebra/
│   │   │   └── linear_algebra.md       # Vectors, matrices, dot product, broadcasting, shapes
│   │   ├── probability_statistics/
│   │   │   └── probability_information_statistics.md  # Probability, entropy, KL, Bayes, Gaussian
│   │   ├── calculus/
│   │   │   ├── calculus.md              # Derivatives, chain rule, gradient descent
│   │   │   └── optimization_algorithms.md  # Batch GD, SGD, Mini-Batch, Momentum, Adam
│   │   ├── activation_functions/
│   │   │   └── activation_functions.md  # Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, SiLU evolution
│   │   └── normalization_regularization/
│   │       └── normalization_dropout_calibration.md  # BatchNorm, LayerNorm, RMSNorm, Dropout, Calibration
│   │
│   ├── 01_ml/                      # ML algorithm deep-dive notes
│   │   ├── 00_universal_ml_notes.md #   Shape rules, complexity, oral drill checklist
│   │   ├── 01_linear_regression.md  #   Linear regression & gradient descent
│   │   ├── 02_logistic_regression.md#   Logistic regression, sigmoid, BCE
│   │   ├── 03_kmeans.md             #   K-Means clustering, WCSS, elbow method
│   │   ├── 04_knn.md               #   KNN, lazy learning, bias-variance via K
│   │   ├── 05_sequence_models.md   #   RNN, LSTM, GRU, Seq2Seq, Attention evolution
│   │   └── 06_decision_trees_random_forest_xgboost_pca.md  # Trees, ensembles, PCA
│   │
│   ├── 02_llm/                      # LLM/Transformer interview notes (21 files)
│   │   ├── 00_LLM Transformer Interview Roadmap.md
│   │   ├── 01_Tokenization & BPE.md
│   │   ├── 02_Embeddings.md
│   │   ├── 03_Transformer Family.md
│   │   ├── 04_Backpropagation.md
│   │   ├── 05_Cross Entropy.md
│   │   ├── 06_Cross Entropy From Logits.md
│   │   ├── 07_Transformer Advanced Topics.md
│   │   ├── 08_RoPE Implementation.md
│   │   ├── 09_GQA & MQA.md
│   │   ├── 10_FlashAttention Intuition.md
│   │   ├── 11_Perplexity, LayerNorm & RMSNorm.md
│   │   ├── 12_KV Cache & Decoding.md
│   │   ├── 13_KV Cache Implementation.md
│   │   ├── 14_RAG.md
│   │   ├── 15_LoRA & Quantization.md
│   │   ├── 16_LoRA Forward Pass.md
│   │   ├── 17_LLM Training Pipeline.md
│   │   ├── 18_Scaling Laws.md
│   │   ├── 19_LLM Evaluation Metrics.md
│   │   ├── 20_MoE (Mixture of Experts).md
│   │   ├── 21_ALiBi, BART, UL2 & SwiGLU.md  #  ALiBi bias, BART denoising, UL2 multi-objective, SwiGLU gated FFN
│   │   ├── 22_Advanced RAG Techniques.md     #  Metadata filtering, parent-child, compression, HyDE, multi-query
│   │   ├── 23_Efficient LLMs.md              #  Continuous batching, streaming, mixed precision, gradient checkpointing, prompt/prefix tuning
│   │   ├── 24_AI Safety & Modern Reasoning.md #  Prompt injection, guardrails, constitutional AI, CoT, test-time compute, reasoning models
│   │   ├── 25_Production ML Data Pipelines.md #  Data collection, validation, feature store, deployment, drift detection, retraining
│   │   └── 26_Beam Search.md                  #  Beam width, log probabilities, length normalization, vs greedy, why not for chat LLMs
│   │
│   ├── 03_system_design/            # ML system design foundations
│   │   ├── 00_system_design_framework.md  # 11-step mental framework
│   │   ├── 01_requirements.md       #   Functional/NFR, ML-specific requirements
│   │   ├── 02_tradeoffs.md          #   Quality/latency/cost, cost of error
│   │   ├── 03_scale_estimation.md   #   GPU memory, serving capacity, storage
│   │   ├── 04_distributed_systems.md#   Training consistency, replication, failure
│   │   └── 05_infrastructure.md     #   GPU clusters, vector DBs, monitoring
│   │
│   ├── 04_cheat_sheets/             # Quick-reference lookup cards (not learning material)
│   │   ├── complexity_table.md         #   One-page complexity lookup
│   │   ├── pattern_keywords.md         #   Keyword → pattern map
│   │   └── ml_coding_checklist.md      #   Last-minute interview checklist
│   │
│   ├── 05_reference/                # Study guides & schedules
│   │   ├── 00_schedule.md           #   12-day intensive prep schedule
│   │   ├── 01_prep_tracker.md       #   Daily ritual tracker & oral drill checklist
│   │   └── 02_data_design.md        #   Dataset design philosophy
│   │
│   └── 06_dsa/                      # DSA & ML coding patterns (learning material)
│       ├── hashmap.md                #   Frequency, lookup, grouping
│       ├── heap.md                   #   Top-K, ranking, priority
│       ├── sliding_window.md         #   Fixed + variable window, chunking
│       ├── binary_search.md          #   Standard, lower/upper bound, answer search
│       ├── two_pointers.md           #   Sorted pair sum, palindrome
│       ├── intervals.md              #   Merge, overlap, meeting rooms
│       ├── dfs_bfs.md                #   Tree/graph traversal templates
│       ├── dfs_trees.md              #   DFS mental model, worked examples
│       ├── retrieval_patterns.md     #   Cosine, top-K, chunking, RAG, metrics
│       ├── pattern_recognition.md    #   Keyword→pattern drills
│       └── interview_framework.md    #   Strategy, edge cases, golden rule
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

1. **Read the notes first** — start with `AI_ENGINEER_ROADMAP.md` for the full 15-module roadmap, then `notes/00_math/` for foundations, then `notes/02_llm/00_LLM Transformer Interview Roadmap.md` for LLM-specific topics, then work through `01`–`20` in order.
2. **Implement from memory** — after reading a note, try to write the corresponding code file from scratch. Compare with the implementation in `code/`.
3. **Oral drill** — use `notes/05_reference/01_prep_tracker.md` for daily oral drill checklists. Explain each concept out loud as if in an interview.
4. **System design** — read `notes/03_system_design/` sessions 0–5 in order, then practice with the mental framework.
5. **DSA patterns** — learn coding patterns in `notes/06_dsa/` (hashmap, heap, sliding window, etc.).
6. **Cheat sheets** — before the interview, skim `notes/04_cheat_sheets/` for quick-reference lookup (complexity table, keyword→pattern map, checklist).

### If you're learning ML/LLM fundamentals

1. Start with `notes/00_math/` — linear algebra, probability, and calculus foundations.
2. Then `code/00_ml/` — these are the building blocks. Each file is self-contained and demonstrates one concept with a tiny, deterministic dataset.
3. Move to `code/01_llm_numpy/` — see how Transformer components work without framework abstractions.
4. Then `code/02_llm_torch/` — see the same concepts as `nn.Module` classes, closer to real-world usage.
5. Read the corresponding `notes/` file for deeper theory, derivations, and interview Q&A.

### If you're reviewing before an interview

- **Day 1–2**: Re-implement all `code/00_ml/` files from memory.
- **Day 3–5**: Re-implement `code/01_llm_numpy/` files from memory.
- **Day 6–7**: Walk through `code/02_llm_torch/` files, explain each line.
- **Day 8–10**: Read all `notes/02_llm/` files, do oral drills.
- **Day 11–12**: Practice system design using `notes/03_system_design/` framework.

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
| `09_swiglu.py` | SwiGLU FFN | Gated FFN (SiLU * gate), LLaMA-style block, RMSNorm, pre-norm, param comparison |

### Notes — Math (`notes/00_math/`)

Foundational math — read before ML/LLM notes.

| Subfolder | File | Topic | Key Concepts |
|---|---|---|---|
| `linear_algebra/` | `linear_algebra.md` | Linear Algebra | Vectors, matrices, dot product, broadcasting, shapes, complexity |
| `probability_statistics/` | `probability_information_statistics.md` | Probability & Statistics | Entropy, cross entropy, KL divergence, mutual information, perplexity, Bayes, Gaussian, variance, correlation |
| `calculus/` | `calculus.md` | Calculus | Derivatives, chain rule, partial derivatives, gradient descent |
| `calculus/` | `optimization_algorithms.md` | Optimization | Batch GD, SGD, Mini-Batch, Momentum, Adam, adaptive learning rates |
| `activation_functions/` | `activation_functions.md` | Activation Functions | Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, SiLU evolution, dying ReLU, smooth gating |
| `normalization_regularization/` | `normalization_dropout_calibration.md` | Norm, Dropout & Calibration | BatchNorm, LayerNorm, RMSNorm, Dropout, inverted dropout, confidence calibration, temperature scaling, label smoothing |

### Notes — ML (`notes/01_ml/`)

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 00 | `00_universal_ml_notes.md` | Universal ML notes | Shape rules, complexity rules, oral drill checklist |
| 01 | `01_linear_regression.md` | Linear regression | MSE, gradient descent, chain rule intuition, learning rate |
| 02 | `02_logistic_regression.md` | Logistic regression | Sigmoid, log-odds, BCE, (p − y) gradient |
| 03 | `03_kmeans.md` | K-Means | WCSS, assignment vs update, elbow method, outlier sensitivity |
| 04 | `04_knn.md` | KNN | Lazy learner, bias-variance via K, complexity |
| 05 | `05_sequence_models.md` | Sequence Models | RNN, LSTM, GRU, Seq2Seq, Attention evolution |
| 06 | `06_decision_trees_random_forest_xgboost_pca.md` | Trees, Ensembles & PCA | Decision trees, Gini/Entropy, Random Forest, XGBoost, PCA, dimensionality reduction |
| 07 | `07_cross_validation_pr_curve_naive_bayes.md` | Final Classical ML | K-fold cross validation, precision-recall curve, Naive Bayes, threshold trade-off |

### Notes — LLM (`notes/02_llm/`)

| # | Topic | L5 Interview Q&A |
|---|-------|------------------|
| 00 | Roadmap | Full topic checklist |
| 01 | Tokenization & BPE | WordPiece, SentencePiece, vocab size tradeoffs |
| 02 | Embeddings | nn.Embedding as lookup table, static vs contextual, why similar words cluster |
| 03 | Transformer Family | Encoder/decoder/cross-attention, BERT vs GPT vs T5, mental model |
| 04 | Backpropagation | Chain rule, computational graph, activation storage, PyTorch autograd |
| 05 | Cross Entropy | KL divergence, label smoothing, focal loss |
| 06 | CE From Logits | Fused implementation, gradient computation, vocab-level CE |
| 07 | Transformer Advanced | ALiBi, pre/post-norm, SwiGLU, sliding window, FlashAttention-2 |
| 08 | RoPE Implementation | Math derivation, interleaved vs half-split, scaling methods |
| 09 | GQA & MQA | KV cache math, quality benchmarks, when to use |
| 10 | FlashAttention | Online softmax, tiling, FA-2/3 improvements |
| 11 | Perplexity, LayerNorm & RMSNorm | Calibration, norm comparisons, gradient analysis |
| 12 | KV Cache & Decoding | PagedAttention, speculative decoding, penalties |
| 13 | KV Cache Implementation | Pre-allocation, multi-layer, GQA-aware, prefix caching |
| 14 | RAG | Hybrid search, reranking, RAGAS, multi-turn, chunking |
| 15 | LoRA & Quantization | QLoRA, rank selection, BF16 vs FP16, quantization methods |
| 16 | LoRA Forward Pass | Initialization, gradient flow, multi-adapter serving |
| 17 | LLM Training Pipeline | Pretraining, SFT, teacher forcing, exposure bias, PPO, DPO, GRPO evolution |
| 18 | Scaling Laws | Chinchilla, overtraining economics, compute-optimal |
| 19 | LLM Evaluation | G-Eval, MT-Bench, AGIEval, data contamination, LLM-as-judge |
| 20 | MoE | DeepSeek MoE, load balancing, capacity factor, MoE serving |
| 21 | ALiBi, BART, UL2 & SwiGLU | ALiBi bias, BART denoising, UL2 multi-objective, SwiGLU gated FFN |
| 22 | Advanced RAG Techniques | Metadata filtering, parent-child, context compression, query expansion, HyDE, multi-query |
| 23 | Efficient LLMs | Continuous batching, streaming, mixed precision, gradient checkpointing, prompt/prefix tuning, PEFT comparison |
| 24 | AI Safety & Modern Reasoning | Prompt injection, guardrails, constitutional AI, chain of thought, test-time compute, reasoning models |
| 25 | Production ML Data Pipelines | Data collection, validation, feature store, deployment strategies, drift detection, retraining |
| 26 | Beam Search | Beam width, log probability scoring, length normalization, vs greedy decoding, why not for chat LLMs |

### Notes — System Design (`notes/03_system_design/`)

| # | File | Topic | L5 Interview Q&A |
|---|------|-------|------------------|
| 00 | `00_system_design_framework.md` | Mental framework | 11-step framework, LLM patterns |
| 01 | `01_requirements.md` | Requirements | Functional/NFR, removal test, ML-specific requirements |
| 02 | `02_tradeoffs.md` | Tradeoffs | Quality/latency/cost/freshness, cost of error, metrics hierarchy |
| 03 | `03_scale_estimation.md` | Scale estimation | GPU memory, LLM serving capacity, vector DB storage, training compute |
| 04 | `04_distributed_systems.md` | Distributed systems | Training consistency, model replication, KV cache, rollouts, failure modes |
| 05 | `05_infrastructure.md` | Infrastructure | GPU clusters, vector DBs, feature stores, model registry, caching, monitoring |

### Notes — Cheat Sheets (`notes/04_cheat_sheets/`)

Quick-reference lookup cards for last-minute review.

| File | Topic |
|------|-------|
| `complexity_table.md` | One-page complexity lookup for all patterns |
| `pattern_keywords.md` | Keyword → pattern map |
| `ml_coding_checklist.md` | Last-minute "can you implement from memory?" checklist |

### Notes — DSA & ML Coding Patterns (`notes/06_dsa/`)

Learning material for coding interview patterns. One topic per file.

| File | Topic |
|------|-------|
| `hashmap.md` | Frequency, lookup, grouping, duplicate detection |
| `heap.md` | Top-K, ranking, priority queue |
| `sliding_window.md` | Fixed + variable window, chunking with overlap |
| `binary_search.md` | Standard, lower/upper bound, binary search on answer |
| `two_pointers.md` | Sorted pair sum, palindrome, merge-style |
| `intervals.md` | Merge, overlap detection, meeting rooms |
| `dfs_bfs.md` | Tree/graph traversal templates, level order |
| `dfs_trees.md` | DFS mental model, 5 worked examples, generic template |
| `retrieval_patterns.md` | Cosine similarity, top-K retrieval, RAG, metrics (Recall@K, MRR, NDCG) |
| `pattern_recognition.md` | Keyword→pattern drills, interview discipline |
| `interview_framework.md` | Strategy, edge cases, golden rule |

---

## Study Schedule

A 12-day intensive plan is in `notes/05_reference/00_schedule.md`. The daily ritual:

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
