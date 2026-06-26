"""LoRA (Low-Rank Adaptation) linear layer in NumPy.

LoRA freezes the original weight matrix W and learns a low-rank update:
    output = x @ W + x @ A @ B

where:
- W: frozen original weights, shape (D_in, D_out)
- A: trainable down-projection, shape (D_in, r)
- B: trainable up-projection,   shape (r, D_out)
- r: rank (much smaller than D_in and D_out)

Parameter reduction:
- Full fine-tune: D_in * D_out parameters
- LoRA: r * (D_in + D_out) parameters
- Example: D=4096, r=8 → 16.7M vs 65K params (256x reduction)

Why LoRA works:
- Fine-tuning updates are empirically low-rank.
- LoRA captures this with A and B while keeping W frozen.
- At inference, can merge: W_merged = W + A @ B (zero overhead).
"""

import numpy as np


def lora_linear(x, W, A, B):
    """LoRA-augmented linear layer forward pass.

    Args:
        x: input,  shape (B, D_in)
        W: frozen base weights, shape (D_in, D_out)
        A: trainable down-projection, shape (D_in, r)
        B: trainable up-projection,   shape (r, D_out)

    Returns:
        Output of shape (B, D_out) = x @ W + x @ A @ B
    """
    # Base output (frozen).
    base = x @ W  # (B, D_out)

    # LoRA adaptation (trainable).
    adapter = x @ A @ B  # (B, r) @ (r, D_out) → (B, D_out)

    return base + adapter


# --- Demo ---
if __name__ == "__main__":
    D_in, D_out, r = 128, 256, 8
    B = 4

    x = np.random.randn(B, D_in)
    W = np.random.randn(D_in, D_out) * 0.01
    A = np.random.randn(D_in, r) * 0.01
    B_param = np.zeros((r, D_out))  # B initialized to zero → starts as identity

    out = lora_linear(x, W, A, B_param)
    print(f"LoRA output shape: {out.shape}")  # (4, 256)

    # Parameter count comparison
    full_params = D_in * D_out
    lora_params = r * (D_in + D_out)
    print(f"Full fine-tune params: {full_params:,}")
    print(f"LoRA params:          {lora_params:,}")
    print(f"Reduction:            {full_params / lora_params:.0f}x")
