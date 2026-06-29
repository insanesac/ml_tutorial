# ALiBi, BART, UL2 & SwiGLU — Deep Dive

## 1. ALiBi (Attention with Linear Biases)

### Motivation

Traditional positional embeddings and even RoPE modify token representations to encode position. ALiBi takes a different approach: **modify the attention scores directly**.

Instead of changing Q and K, it adds a distance-based linear bias before the softmax.

```
Normal Attention:  Score = Q @ K^T / √d_k
ALiBi:             Score = Q @ K^T / √d_k + Bias

where  Bias = -m × distance
  m        = slope assigned to each attention head
  distance = number of tokens apart
```

### Example

Predicting token 5:

```
Token5 → Token5:  distance = 0  →  bias = 0
Token5 → Token4:  distance = 1  →  bias = -m
Token5 → Token3:  distance = 2  →  bias = -2m
Token5 → Token2:  distance = 3  →  bias = -3m
```

Farther tokens receive a larger negative penalty before softmax.

### Why Different Slopes?

Each attention head receives a different slope:

| Slope | Penalty | Specialization |
|---|---|---|
| Small | Small | Better long-range attention |
| Large | Large | Better local attention |

This naturally creates **head specialization** across attention heads.

### Advantages

- Extremely simple implementation
- No learned positional embeddings
- Excellent extrapolation to longer sequence lengths
- Minimal computational overhead

### Implementation

```python
scores = Q @ K.transpose(-2, -1)
scores /= math.sqrt(d_k)
scores += alibi_bias  # precomputed distance-based bias
weights = softmax(scores)
```

### ALiBi vs RoPE

| | RoPE | ALiBi |
|---|---|---|
| Mechanism | Rotates Q & K | Adds score bias |
| Encoding | Relative positional encoding | Relative positional bias |
| Compute | Slightly more | Very cheap |
| Extrapolation | Good (with scaling) | Excellent |
| Adoption | Widely used (LLaMA, Gemma) | BLOOM |

---

## 2. BART (Bidirectional and Auto-Regressive Transformers)

### Architecture

BART combines:
- **BERT-style Encoder** (Bidirectional)
- **GPT-style Decoder** (Autoregressive)

```
Corrupted Input
  ↓
Bidirectional Encoder
  ↓
Autoregressive Decoder
  ↓
Original Sentence
```

### Core Idea

Instead of predicting the next token or masked words, BART **corrupts** an input sentence and trains the model to **reconstruct** the original.

```
Original:   "The cat sat on the mat."
Corrupted:  "The cat <mask> the mat."
Target:     "The cat sat on the mat."
```

### Types of Corruption

| Corruption | Example | What It Teaches |
|---|---|---|
| **Token Masking** | `The cat <mask> the mat.` | Basic word prediction |
| **Span Masking** | `The <mask> mat.` | Syntax, semantics, long-range dependencies |
| **Sentence Permutation** | `Sentence C, A, B` | Discourse structure |
| **Token Deletion** | `The cat on mat.` | Grammar reconstruction |

### Why Span Masking?

Unlike masking individual words, span masking forces the decoder to **generate multiple coherent tokens**. This better prepares the model for summarization, translation, grammar correction, and paraphrasing.

### Applications

- Text Summarization
- Machine Translation
- Question Answering
- Text Correction
- Paraphrasing

### GPT vs BERT vs BART

| | GPT | BERT | BART |
|---|---|---|---|
| Architecture | Decoder only | Encoder only | Encoder + Decoder |
| Objective | Next Token Prediction | Masked Token Prediction | Denoising Reconstruction |
| Strength | Text Generation | Understanding | Understanding + Generation |

---

## 3. UL2 (Unifying Language Learning)

### Motivation

Instead of training with only one objective like GPT or BART, UL2 trains using **multiple objectives simultaneously**.

One model learns: language understanding, reconstruction, and long-form generation.

### Three Training Objectives

| Objective | Description | Similar To |
|---|---|---|
| **R-Denoiser** | Regular denoising — recover missing text | BART |
| **S-Denoiser** | Short span masking — many small masks | BERT |
| **X-Denoiser** | Extreme denoising — minimal context, generate long output | Novel |

### X-Denoiser Example

```
Input:   "The <mask>"
Target:  "The cat sat on the mat while watching birds outside."
```

Most of the sentence is removed. The model must generate long coherent sequences from minimal context.

### Why Extreme Denoising?

Small masks leave plenty of context. Extreme denoising forces the model to rely on:
- World knowledge
- Grammar
- Long-range reasoning
- Coherent sequence generation

### Architecture

Standard Encoder-Decoder Transformer. The innovation lies in the **training objective**, not the architecture.

### GPT vs BART vs UL2

| | GPT | BART | UL2 |
|---|---|---|---|
| Objectives | One | One | Multiple |
| Training | Next Token Prediction | Denoising | Mixture of Denoising |
| Architecture | Decoder only | Encoder-Decoder | Encoder-Decoder |

---

## 4. SwiGLU

### Motivation

Original Transformer FFN:

```
Linear → GELU → Linear
```

Modern LLMs replace this with SwiGLU — a **gated** feed-forward network.

### Core Idea

Instead of one projection, use two:

```
         x
       /   \
   Linear    Linear
     |         |
    SiLU     Value
     \       /
  Element-wise ×
       |
   Output Linear
```

One branch creates **values**. The other branch acts as a **learned gate**.

### Mathematical Form

```
Instead of:  GELU(Wx)

SwiGLU:      SiLU(W₁x) ⊙ (W₂x)  →  W₃(...)

where ⊙ = element-wise multiplication
```

### Why Gating?

The gate decides how much information should pass through:

```
Value = 8, Gate = 0.9   →  Output = 7.2   (important feature amplified)
Value = 8, Gate = 0.05  →  Output = 0.4   (less useful feature suppressed)
```

### PyTorch Implementation

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)  # gate
        self.w2 = nn.Linear(d_model, hidden_dim)  # value
        self.w3 = nn.Linear(hidden_dim, d_model)  # output

    def forward(self, x):
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)
```

### GELU vs SwiGLU

| | GELU | SwiGLU |
|---|---|---|
| Projections | One | Two |
| Activation | GELU | SiLU + Learned Gate |
| Gating | No | Explicit |
| Used in | Original Transformer | Modern LLMs (LLaMA, Mistral, PaLM) |
| Parameters | 2 matrices | 3 matrices (50% more, offset by ~2.67x expansion vs 4x) |

---

## Key Interview Takeaways

- **ALiBi:** Adds linear distance-based bias to attention scores. Different slopes per head create local/long-range specialization. Excellent length extrapolation.
- **BART:** Encoder-decoder Transformer trained by reconstructing corrupted text (denoising autoencoder). Strong for summarization and translation.
- **UL2:** Google's multi-objective pretraining framework combining R/S/X-denoising. Produces versatile models for understanding + generation.
- **SwiGLU:** Gated FFN replacing GELU in modern LLMs. SiLU gate × value, with 3 weight matrices. Used by LLaMA, Mistral, PaLM.
