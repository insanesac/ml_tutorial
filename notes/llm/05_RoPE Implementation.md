# RoPE Implementation (Rotary Positional Embeddings)

## Why RoPE?

Attention itself has no order information.

RoPE injects position by rotating Q and K vectors with position-dependent angles.

Compared to absolute positional embeddings, RoPE often extrapolates better to longer contexts.

## Core Idea

For each token position `pos`, rotate 2D feature pairs `(x_even, x_odd)`:

`[x_even', x_odd'] = rotation(theta_pos) * [x_even, x_odd]`

The relative angle difference between positions encodes relative position.

## Shape Setup

- `Q, K`: `(B, H, N, D_head)`
- RoPE is applied on last dim `D_head` (usually even).

## Minimal NumPy Implementation

```python
def apply_rope(x):
    # x: (B, H, N, D)
    B, H, N, D = x.shape
    assert D % 2 == 0

    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    # frequencies
    idx = np.arange(half)
    inv_freq = 1.0 / (10000 ** (idx / half))

    # positions
    pos = np.arange(N)
    theta = np.outer(pos, inv_freq)           # (N, half)

    cos_t = np.cos(theta)[None, None, :, :]   # (1,1,N,half)
    sin_t = np.sin(theta)[None, None, :, :]

    # rotary transform
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t

    return np.concatenate([out1, out2], axis=-1)
```

## Where to Apply

Apply RoPE to **Q and K**, not V.

```python
Q = apply_rope(Q)
K = apply_rope(K)
```

Then continue normal attention.

## Complexity

- Time: `O(B * H * N * D_head)`
- Space: same order as Q/K tensors

## Common Traps

- Forgetting `D_head` must be even
- Applying RoPE to V
- Wrong broadcast shape for `cos/sin`

## Mathematical Derivation

### 2D Case (Single Pair)

For a 2D vector `[x1, x2]` at position `pos`, RoPE applies a rotation:

```
[x1']   [cos(θ_pos)  -sin(θ_pos)] [x1]
[x2'] = [sin(θ_pos)   cos(θ_pos)] [x2]
```

Where `θ_pos = pos * inv_freq` and `inv_freq = 1 / base^(2i / D)` for dimension pair `i`.

### Why This Encodes Relative Position

When query at position `m` attends to key at position `n`:

```
Q_m · K_n = (R_m · q) · (R_n · k)
          = q · (R_m^T · R_n) · k
          = q · R_{n-m} · k
```

The dot product depends on `R_{n-m}` — a rotation by the **relative** angle `(n-m) * θ`. This is why RoPE naturally encodes relative position.

### General D-Dimensional Case

For `D` dimensions, split into `D/2` pairs and rotate each pair independently with different frequencies:

```
θ_i = pos / base^(2i / D)    for i = 0, 1, ..., D/2 - 1
```

- Low-frequency pairs (small `i`): slow rotation, captures long-range position
- High-frequency pairs (large `i`): fast rotation, captures short-range position
- `base` is typically 10000 (same as sinusoidal encoding)

### Frequency Spectrum

```
i=0:     θ = pos / 1           (highest freq — every position is distinct)
i=D/4:   θ = pos / 100          (medium freq)
i=D/2-1: θ = pos / 10000        (lowest freq — slowly changing)
```

This multi-frequency design lets RoPE capture both local and global position simultaneously.

---

## Interleaved vs Half-Split (GPT-NeoX vs LLaMA Style)

### Interleaved (Original RoPE, GPT-J style)

Pairs are `(x[0], x[1]), (x[2], x[3]), ...`:

```
x = [x0, x1, x2, x3, x4, x5, ...]
pairs: (x0, x1), (x2, x3), (x4, x5), ...
```

### Half-Split (LLaMA / HuggingFace style)

Pairs are `(x[0], x[D/2]), (x[1], x[D/2+1]), ...`:

```
x = [x0, x1, ..., x_{D/2-1}, x_{D/2}, x_{D/2+1}, ...]
pairs: (x0, x_{D/2}), (x1, x_{D/2+1}), ...
```

### Why This Matters

- Both are mathematically equivalent — just different memory layouts.
- **If you mix them up, the model breaks silently** — outputs look like garbage.
- HuggingFace uses half-split; some implementations use interleaved.
- Always check which convention your model uses.

### HuggingFace Style Implementation (Half-Split)

```python
def apply_rope_hf_style(x):
    # x: (B, H, N, D)
    B, H, N, D = x.shape
    half = D // 2

    # Frequencies
    inv_freq = 1.0 / (10000 ** (np.arange(0, D, 2) / D))  # (D/2,)
    pos = np.arange(N)  # (N,)
    theta = np.outer(pos, inv_freq)  # (N, D/2)

    cos = np.cos(theta)  # (N, D/2)
    sin = np.sin(theta)  # (N, D/2)

    # Repeat to full dimension: [cos, cos] and [sin, sin]
    cos = np.concatenate([cos, cos], axis=-1)  # (N, D)
    sin = np.concatenate([sin, sin], axis=-1)  # (N, D)

    # Reshape for broadcasting
    cos = cos[None, None, :, :]  # (1, 1, N, D)
    sin = sin[None, None, :, :]

    # Rotate: x_rotated = x * cos + rotate_half(x) * sin
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotate_half = np.concatenate([-x2, x1], axis=-1)

    return x * cos + rotate_half * sin
```

---

## RoPE Scaling for Long Context (L5 Critical)

### The Problem

RoPE is trained with a maximum position `N_train`. Beyond this, the rotation angles become very large, and the model's attention degrades because it never saw those angles during training.

### Method 1: Position Interpolation (PI)

Scale positions down to fit within the training range:

```
pos_scaled = pos * (N_train / N_target)
```

- Simple, but degrades quality on short contexts (everything is compressed).
- Used in: early LLaMA long-context extensions.

### Method 2: NTK-Aware Scaling

Scale the base frequency instead of positions:

```
base_scaled = base * (scale_factor)^(D / (D - 2))
```

- Preserves high-frequency resolution (short-range position still works).
- Stretches low frequencies for long-range.
- Better than PI for maintaining short-context quality.

### Method 3: YaRN (Yet another RoPE extensioN)

Combines NTK-aware scaling with attention temperature adjustment:

```
1. NTK-aware frequency scaling
2. Scale attention scores by a temperature factor for out-of-range positions
3. Fine-tune on a small amount of long-context data
```

- Best performing RoPE extension method.
- Used in: Mistral long-context, some LLaMA derivatives.

### Method 4: Dynamic NTK

Adjust the scaling factor dynamically based on the actual sequence length:

```
scale = max(1, current_seq_len / N_train)
```

- No need to predefine the target length.
- Used in: some serving frameworks.

### Comparison

| Method | Short Context | Long Context | Needs Fine-tuning? |
|---|---|---|---|
| Position Interpolation | Degraded | OK | Light |
| NTK-Aware | Preserved | Good | No |
| YaRN | Preserved | Best | Light |
| Dynamic NTK | Preserved | Good | No |

---

## PyTorch Implementation (Production Style)

```python
import torch

def precompute_rope_cos_sin(seq_len, dim, base=10000, device='cpu'):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(seq_len, device=device).float()
    theta = torch.outer(pos, inv_freq)  # (seq_len, dim/2)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    # Stack and repeat for full dimension
    cos = torch.cat([cos, cos], dim=-1)  # (seq_len, dim)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin

def apply_rope(x, cos, sin):
    # x: (B, H, N, D)
    # cos, sin: (N, D) or (1, 1, N, D)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    # cos/sin shape: (1, 1, N, D) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half * sin
```

### Precomputation Optimization

Cos/sin tables are **precomputed once** and cached — they don't change during training or inference. This makes RoPE essentially free at runtime (just element-wise multiply).

---

## L5 Interview Q&A

### Q: "Why does RoPE encode relative position when it uses absolute positions?"

The rotation applied to position `m` is `R(m*θ)`. When Q at position `m` dot-products with K at position `n`:

```
Q_m · K_n = q^T · R_m^T · R_n · k = q^T · R_{n-m} · k
```

The result depends on `n - m` (relative position), not on `m` or `n` individually. The absolute position is used to compute the rotation, but the attention dot product only sees the **difference**.

### Q: "Why not apply RoPE to V?"

RoPE's purpose is to encode position into the **attention scores** (`Q · K`). The value `V` is what gets aggregated after attention weights are computed — it doesn't participate in the score computation. Rotating V would change the output values without affecting which tokens attend to which, providing no positional benefit.

### Q: "How would you extend a model trained with 4K context to 128K?"

1. **RoPE scaling**: Use NTK-aware or YaRN scaling to stretch the frequency spectrum.
2. **Fine-tuning**: Train on a small amount of long-context data (a few hundred million tokens at 128K).
3. **FlashAttention**: Essential for memory — 128K attention matrix in FP16 is 32GB per head without FlashAttention.
4. **Sliding window + global**: Hybrid attention (local window + a few global tokens) can reduce memory.
5. **RAG as alternative**: If the use case doesn't need full 128K reasoning, RAG is cheaper and more reliable.

### Q: "What happens if you use the wrong RoPE convention (interleaved vs half-split)?"

The model produces garbage output. The rotation is applied to the wrong pairs, so the positional encoding is completely wrong. This is a **silent failure** — no error is raised, but the model's outputs are nonsensical. Always verify the convention matches the checkpoint.

### Q: "What's the memory cost of RoPE?"

- **Parameters**: 0 (no learnable parameters).
- **Runtime**: cos/sin tables of shape `(max_seq_len, D)` — for 128K context and D=128, this is `128K * 128 * 4 bytes * 2 = 128MB`. Precomputed once, cached forever.
- **Compute**: Two element-wise multiplications per Q/K tensor — negligible compared to the matmul.

---

## Interview Sound Bites

- RoPE encodes position by rotating Q/K vectors in 2D feature pairs — the dot product depends only on relative position.
- Multi-frequency design: low frequencies for long-range, high frequencies for short-range.
- Apply to Q and K, never V — V doesn't participate in attention scores.
- Two conventions (interleaved vs half-split) — mixing them causes silent failure.
- RoPE scaling (NTK, YaRN) extends context window by stretching the frequency spectrum.
- No learnable parameters — cos/sin tables are precomputed and cached.
- Used by LLaMA, Gemma, Mistral, and most modern decoder-only LLMs.
