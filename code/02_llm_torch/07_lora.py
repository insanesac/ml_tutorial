"""LoRA (Low-Rank Adaptation) linear layer in PyTorch.

LoRA freezes the original weight matrix W and learns a low-rank update:
    output = x @ W + x @ A @ B

where:
- W: frozen original weights, shape (D_in, D_out)
- A: trainable down-projection, shape (D_in, r)
- B: trainable up-projection,   shape (r, D_out)
- r: rank (much smaller than D_in and D_out)

Initialization:
- A: initialized with Kaiming uniform (standard linear init).
- B: initialized to ZERO → adapter starts as identity (no change at init).
  This ensures the model behaves exactly like the base model before training.

At inference, can merge: W_merged = W + A @ B (zero overhead, no extra compute).
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA low-rank adaptation.

    Args:
        in_features:  input dimension D_in.
        out_features: output dimension D_out.
        r:            LoRA rank (typically 4-64).
        bias:         whether to include bias in base layer.
    """

    def __init__(self, in_features, out_features, r=8, bias=True):
        super().__init__()

        # Base linear layer (frozen during LoRA training).
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA down-projection: D_in → r
        self.lora_A = nn.Linear(in_features, r, bias=False)

        # LoRA up-projection: r → D_out
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Initialize A with Kaiming, B with zero.
        # This ensures adapter output = 0 at init → base model behavior preserved.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """Forward pass: base(x) + lora_B(lora_A(x)).

        Args:
            x: input, shape (..., in_features).

        Returns:
            Output of shape (..., out_features).
        """
        # Base output (frozen).
        base_out = self.base(x)

        # LoRA adaptation (trainable): x → A → B.
        lora_out = self.lora_B(self.lora_A(x))

        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into base layer for zero-overhead inference.

        After merging, the adapter is folded into base.weight:
            W_merged = W + B @ A
        """
        with torch.no_grad():
            # B.weight: (out_features, r), A.weight: (r, in_features)
            # B @ A: (out_features, in_features) — same shape as base.weight
            self.base.weight += self.lora_B.weight @ self.lora_A.weight


# --- Demo ---
if __name__ == "__main__":
    D_in, D_out, r = 128, 256, 8

    layer = LoRALinear(D_in, D_out, r=r)

    x = torch.randn(4, D_in)
    out = layer(x)
    print(f"Output shape: {out.shape}")  # (4, 256)

    # Parameter count
    base_params = D_in * D_out
    lora_params = r * (D_in + D_out)
    print(f"Base params: {base_params:,} (frozen)")
    print(f"LoRA params: {lora_params:,} (trainable)")
    print(f"Reduction:   {base_params / lora_params:.0f}x")

    # Merge for inference
    layer.merge_weights()
    print("Weights merged — adapter has zero inference overhead now.")
