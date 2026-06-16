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

## Interview Sound Bites

- RoPE encodes position by rotating Q/K vectors in feature pairs.
- It naturally supports relative position behavior.
- It is used by LLaMA/Gemma/Mistral-style models.
