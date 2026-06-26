# ML Interview Prep

Google L5 ML/LLM interview preparation repository.

## Structure

```
algo/
├── code/
│   ├── ml/                      # ML algorithm implementations (NumPy)
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── kmeans.py
│   │   ├── softmax.py
│   │   ├── cross_entropy.py
│   │   ├── neural_network.py
│   │   ├── backpropagation.py
│   │   ├── relu.py
│   │   ├── metrics.py
│   │   ├── bias_variance.py
│   │   ├── regularization.py
│   │   └── data_splits.py
│   ├── llm/                     # LLM/Transformer implementations
│   │   ├── attention.py         # Scaled dot-product attention
│   │   ├── multihead_attention.py
│   │   ├── transformer_numpy.py # Full NumPy transformer (educational)
│   │   ├── transformer_utils.py # RoPE, KV cache utilities
│   │   ├── torch_implementation.py  # PyTorch transformer block
│   │   ├── gpt_model.py         # GPT model with weight tying
│   │   ├── gqa.py               # Grouped-Query Attention (NumPy)
│   │   ├── gqa_torch.py         # GQA (PyTorch)
│   │   ├── mqa.py               # Multi-Query Attention
│   │   ├── rope.py              # Rotary Positional Embeddings (NumPy)
│   │   ├── rope_torch.py        # RoPE (PyTorch)
│   │   ├── softmax.py
│   │   ├── cosine_similarity.py
│   │   ├── rlhf.py              # PPO, DPO, GRPO loss functions
│   │   └── llm_basics.py        # Softmax, CE, perplexity, norms, sampling
│   └── practice/                # Coding practice & trials
│       ├── practice.py
│       ├── trials.py
│       └── test.py
├── notes/
│   ├── ml/                      # ML algorithm notes (Volumes 1-4)
│   ├── llm/                     # LLM interview notes (00-17)
│   ├── system_design/           # System design foundations & framework
│   ├── cheat_sheets/            # Interview pattern cheat sheets
│   └── reference/               # Tracker, schedule, dataset design
└── .venv/
```

## Key Topics Covered

### ML Algorithms
Linear/logistic regression, KNN, K-means, softmax, cross-entropy, neural networks, backpropagation, metrics, regularization, bias-variance.

### LLM/Transformers
Tokenization, attention (MHA/GQA/MQA), RoPE, FlashAttention, KV cache, decoding strategies, perplexity, LayerNorm/RMSNorm, RAG, LoRA/QLoRA, RLHF/DPO/GRPO, scaling laws, evaluation metrics, MoE.

### System Design
ML system design framework, functional/non-functional requirements, tradeoffs, scale estimation, distributed systems, infrastructure components, LLM serving architecture.

## Daily Practice
See `notes/reference/ML_Interview_Prep_Tracker.md` and `notes/reference/SCHEDULE.md`.
