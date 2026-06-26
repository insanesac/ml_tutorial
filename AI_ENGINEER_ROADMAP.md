# AI Engineer Roadmap (Senior / L5+)

## Goal

Build deep, interview-ready knowledge across modern AI, Machine Learning, LLMs, and Production AI Systems.

Each topic should eventually include:
- Intuition (Why does it exist?)
- Problem it solves
- Mathematical foundation
- PyTorch implementation
- Production considerations
- Common interview questions

---

## Module 1 — Mathematics & Optimization ⭐⭐⭐⭐⭐

### Linear Algebra
- Scalars, Vectors, Matrices, Tensors
- Matrix Multiplication
- Dot Product
- Matrix Transpose
- Matrix Shapes
- Broadcasting

### Probability & Statistics
- Probability
- Conditional Probability
- Bayes Theorem
- Expectation
- Variance
- Covariance
- Gaussian Distribution

### Information Theory
- Entropy
- Cross Entropy
- KL Divergence
- Mutual Information
- Perplexity

### Optimization
- Gradient Descent
- Backpropagation
- Chain Rule
- SGD
- Momentum
- RMSProp
- Adam
- Learning Rate Scheduling
- Gradient Clipping
- Weight Decay

---

## Module 2 — Deep Learning Foundations ⭐⭐⭐⭐⭐

- Perceptron
- Multi-Layer Perceptron (MLP)
- Activation Functions: ReLU, GELU, SiLU/Swish, Sigmoid, Tanh
- Loss Functions: MSE, BCE, Cross Entropy
- Batch Normalization
- LayerNorm
- RMSNorm
- Residual Connections / Skip Connections
- Weight Initialization
- Dropout

---

## Module 3 — Classical Machine Learning ⭐⭐⭐⭐

### Supervised Learning
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- SVM
- Naive Bayes
- KNN

### Unsupervised Learning
- K-Means
- DBSCAN
- Hierarchical Clustering
- PCA
- t-SNE
- UMAP

### Evaluation
- Precision, Recall, F1
- ROC-AUC, PR Curve
- Confusion Matrix
- Bias vs Variance
- Cross Validation

---

## Module 4 — NLP Foundations ⭐⭐⭐⭐⭐

### Tokenization
- Character Tokenization
- Word Tokenization
- BPE
- WordPiece
- SentencePiece
- Unigram Language Model
- Special Tokens
- Vocabulary Construction
- OOV Handling

### Word Embeddings
- One-Hot Encoding
- Word2Vec
- GloVe
- FastText
- Contextual Embeddings

### Sequence Models
- RNN
- LSTM
- GRU
- Seq2Seq
- Attention

---

## Module 5 — Transformers & LLMs ⭐⭐⭐⭐⭐

### Architecture
- Self Attention
- Cross Attention
- Multi-Head Attention
- MQA
- GQA
- Transformer Block
- Feed Forward Networks
- SwiGLU

### Positional Encoding
- Sinusoidal
- Learned Embeddings
- RoPE
- ALiBi

### Transformer Variants
- GPT
- BERT
- T5
- BART
- UL2

### Decoder Optimizations
- KV Cache
- Flash Attention
- Sliding Window Attention
- Sparse Attention
- Paged Attention

---

## Module 6 — LLM Training ⭐⭐⭐⭐⭐

- Next Token Prediction
- Masked Language Modeling
- Teacher Forcing
- Instruction Tuning
- Supervised Fine-Tuning (SFT)
- RLHF
- PPO
- DPO
- GRPO
- Reward Models
- Preference Optimization

---

## Module 7 — LLM Inference ⭐⭐⭐⭐⭐

- Greedy Decoding
- Temperature Sampling
- Top-K
- Top-P (Nucleus Sampling)
- Beam Search
- Speculative Decoding
- Continuous Batching
- Streaming Generation

---

## Module 8 — Efficient LLMs ⭐⭐⭐⭐⭐

- Quantization: FP16, BF16, INT8, INT4
- LoRA / QLoRA
- Prefix Tuning
- Prompt Tuning
- Adapter Layers
- Model Offloading
- CPU Offloading
- Disk Offloading
- Gradient Checkpointing
- Mixed Precision Training

---

## Module 9 — Retrieval-Augmented Generation (RAG) ⭐⭐⭐⭐⭐

### Pipeline
- Chunking Strategies
- Embedding Models
- Vector Databases
- BM25
- Hybrid Search
- Metadata Filtering
- Parent-Child Retrieval
- Context Compression
- Query Expansion
- HyDE
- Multi-Query Retrieval
- Re-ranking
- Knowledge Graph RAG

### Evaluation
- Recall@K
- Precision@K
- MRR
- NDCG
- Groundedness
- Hallucination Rate
- Citation Accuracy

---

## Module 10 — AI Agents ⭐⭐⭐⭐⭐

- Function Calling
- Tool Use
- Planning
- Reflection
- Memory
- ReAct
- MCP
- A2A
- Multi-Agent Systems

---

## Module 11 — Computer Vision ⭐⭐⭐⭐

- CNNs
- ResNet
- EfficientNet
- Vision Transformers
- YOLO
- DETR
- CLIP
- SAM
- OCR
- Image Segmentation

---

## Module 12 — Multimodal AI ⭐⭐⭐⭐

- CLIP
- BLIP
- LLaVA
- Qwen-VL
- Gemma Vision
- Audio Models
- Speech Models
- Video Understanding
- Diffusion Models

---

## Module 13 — Production ML Systems ⭐⭐⭐⭐⭐

### Training Systems
- Data Pipelines
- Distributed Training
- Data Parallelism
- Tensor Parallelism
- Pipeline Parallelism
- FSDP
- DeepSpeed

### Serving
- vLLM
- TensorRT-LLM
- Triton
- Continuous Batching
- Model Caching
- Load Balancing

### MLOps
- Feature Store
- Model Registry
- Experiment Tracking
- Drift Detection
- Retraining
- A/B Testing
- Shadow Deployment
- Canary Releases

---

## Module 14 — AI Safety & Alignment ⭐⭐⭐⭐

- Prompt Injection
- Jailbreaks
- Guardrails
- PII Detection
- Red Teaming
- Alignment
- Hallucination Mitigation
- Constitutional AI

---

## Module 15 — Advanced AI Research ⭐⭐⭐

- Scaling Laws
- Chain of Thought
- Tree of Thoughts
- Test-Time Compute
- Reasoning Models
- Synthetic Data
- Self-Play
- World Models
- Diffusion Transformers

---

## Interview Coding Topics

- NumPy Fundamentals
- PyTorch Fundamentals
- Tensor Shapes
- Attention Implementation
- Transformer Implementation
- Tokenizers
- Gradient Computation
- Backpropagation
- Loss Functions
- Retrieval Algorithms
- Beam Search
- Top-K / Top-P
- Trie
- Heap
- Sliding Window
- Dynamic Programming
- Trees & Graphs

---

## Current Learning Order

### Phase 1 (Current)
- Backpropagation
- Tokenization
- WordPiece
- SentencePiece
- BERT
- GPT
- Encoder vs Decoder
- PyTorch Autograd
- Flash Attention
- MoE
- Speculative Decoding
- Distributed Training Basics

### Phase 2
- Classical ML Refresh
- Vision
- Multimodal
- Production ML
- Distributed Systems

### Phase 3
- Research Papers
- CUDA Internals
- Advanced Transformer Optimizations
- Inference Engine Internals
- Open Source LLM Implementations
