"""SwiGLU Feed-Forward Network in PyTorch.

SwiGLU (Swish-Gated Linear Unit) replaces the standard ReLU FFN in modern
transformers (LLaMA, PaLM, Gemma). It completes the transformer architecture:

    Standard FFN:  Linear → ReLU → Linear         (2 weight matrices)
    SwiGLU FFN:    Linear → Swish * Linear_gate → Linear  (3 weight matrices)

Formula:
    FFN_SwiGLU(x) = (SiLU(xW_gate) * xW_up) W_down

where SiLU(x) = x * sigmoid(x)  (also called Swish).

The hidden dimension is typically (2/3) * 4 * d_model, rounded to a multiple
of 64, to keep the parameter count comparable to the standard 4x FFN.

Shapes (input X of shape (B, N, D)):
- W_gate: (D, d_ff)    gate path
- W_up:   (D, d_ff)    up projection
- W_down: (d_ff, D)    down projection
- Output: (B, N, D)

Why SwiGLU over ReLU?
- Gating allows the network to selectively pass or suppress features
- SiLU is smooth (non-zero gradient everywhere), unlike ReLU's dead neurons
- Empirically better performance at scale (PaLM, LLaMA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    Args:
        d_model:  model embedding dimension D.
        d_ff:     hidden dimension. Default: round(2/3 * 4 * d_model).
                  Rounded to nearest multiple of 64 for hardware efficiency.
    """

    def __init__(self, d_model, d_ff=None):
        super().__init__()

        if d_ff is None:
            # (2/3) * 4 * D, rounded to nearest multiple of 64
            d_ff = int((2 / 3) * 4 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64

        # Gate projection: D → d_ff
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        # Up projection: D → d_ff
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        # Down projection: d_ff → D
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

        self.d_ff = d_ff

    def forward(self, x):
        """SwiGLU forward pass.

        Args:
            x: input tensor, shape (B, N, D).

        Returns:
            Output tensor, shape (B, N, D).

        Step-by-step:
            1. gate = x @ W_gate   → (B, N, d_ff)
            2. up   = x @ W_up     → (B, N, d_ff)
            3. activated = SiLU(gate) * up   → (B, N, d_ff)
            4. out = activated @ W_down      → (B, N, D)
        """
        gate = self.w_gate(x)           # (B, N, d_ff)
        up = self.w_up(x)               # (B, N, d_ff)
        activated = F.silu(gate) * up   # SiLU(x) = x * sigmoid(x)
        return self.w_down(activated)   # (B, N, D)


class SwiGLUTransformerBlock(nn.Module):
    """Transformer block with SwiGLU FFN (pre-norm style, like LLaMA).

    Uses RMSNorm instead of LayerNorm, matching modern LLM conventions.

    Args:
        d_model:    model embedding dimension D.
        num_heads:  number of attention heads H.
        d_ff:       SwiGLU hidden dimension (default: auto-computed).
    """

    def __init__(self, d_model, num_heads, d_ff=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(d_model, d_ff)

        # RMSNorm (pre-norm)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def attention(self, x, mask=None):
        """Multi-head self-attention (pre-norm input already applied)."""
        B, N, D = x.shape
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(2, 3)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)

    def forward(self, x, mask=None):
        """Pre-norm transformer block: norm → sublayer → residual."""
        # Attention sub-layer
        x = x + self.attention(self.norm1(x), mask)

        # SwiGLU FFN sub-layer
        x = x + self.ffn(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # --- SwiGLU FFN standalone ---
    print("=" * 60)
    print("SwiGLU FFN")
    print("=" * 60)

    B, N, D = 2, 8, 64
    x = torch.randn(B, N, D)

    ffn = SwiGLUFFN(D)
    out = ffn(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Hidden dim:   {ffn.d_ff}  (2/3 * 4 * {D} = {int(2/3 * 4 * D)}, rounded to {ffn.d_ff})")
    print(f"Param count:  {sum(p.numel() for p in ffn.parameters()):,}")
    print()

    # --- Compare with standard ReLU FFN ---
    print("Comparison: SwiGLU vs Standard ReLU FFN")
    print("-" * 40)

    std_ffn = nn.Sequential(
        nn.Linear(D, 4 * D),
        nn.ReLU(),
        nn.Linear(4 * D, D),
    )
    std_params = sum(p.numel() for p in std_ffn.parameters())
    swiglu_params = sum(p.numel() for p in ffn.parameters())

    print(f"Standard FFN params:  {std_params:,}  (2 matrices: {D}×{4*D} + {4*D}×{D})")
    print(f"SwiGLU FFN params:    {swiglu_params:,}  (3 matrices: {D}×{ffn.d_ff} ×3... + {ffn.d_ff}×{D})")
    print()

    # --- SwiGLU Transformer Block ---
    print("=" * 60)
    print("SwiGLU Transformer Block (LLaMA-style, pre-norm + RMSNorm)")
    print("=" * 60)

    block = SwiGLUTransformerBlock(d_model=64, num_heads=8)
    out_block = block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out_block.shape}")
    print(f"Param count:  {sum(p.numel() for p in block.parameters()):,}")
    print()

    # --- Gradient check ---
    print("Gradient check:")
    out_block.sum().backward()
    grad_ok = all(p.grad is not None for p in block.parameters())
    print(f"  All parameters have gradients: {grad_ok}")
    print()

    # --- SiLU visualization ---
    print("SiLU(x) = x * sigmoid(x)  (a.k.a. Swish)")
    print("-" * 40)
    sample = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    silu_vals = F.silu(sample)
    relu_vals = F.relu(sample)
    print(f"{'x':>6} | {'SiLU(x)':>10} | {'ReLU(x)':>10}")
    print("-" * 32)
    for x_val, s_val, r_val in zip(sample, silu_vals, relu_vals):
        print(f"{x_val:6.1f} | {s_val:10.4f} | {r_val:10.4f}")
    print()
    print("Key difference: SiLU has non-zero gradient for x < 0 (smooth gating),")
    print("while ReLU kills gradients entirely for negative inputs.")
