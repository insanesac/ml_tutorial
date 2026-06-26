"""KV Cache for autoregressive decoding in PyTorch.

During autoregressive generation, each new token only needs:
- Q for the new token (computed fresh each step)
- K and V for ALL tokens (including all previous ones)

Without cache: recompute K, V for all previous tokens every step → O(N^2) total.
With cache:    store previous K, V, only compute K, V for the new token → O(N) total.

Cache shape: (B, H, T, head_dim) where T grows by 1 each step.
New K/V shape: (B, H, 1, head_dim) — just the latest token.
"""

import torch
import torch.nn as nn


class CacheKV(nn.Module):
    """KV Cache module for autoregressive inference.

    Stores K and V tensors across decoding steps.
    On first call, initializes cache with the provided K, V.
    On subsequent calls, appends new K, V along the sequence dimension.
    """

    def __init__(self):
        super().__init__()
        self.K_cache = None
        self.V_cache = None

    def update(self, K_new, V_new):
        """Append new K, V vectors to the cache.

        Args:
            K_new: new keys,   shape (B, H, 1, head_dim)
            V_new: new values, shape (B, H, 1, head_dim)

        Returns:
            Updated K_cache, V_cache — shape (B, H, T, head_dim)
        """
        if self.K_cache is None:
            # First step: initialize cache.
            self.K_cache = K_new
            self.V_cache = V_new
        else:
            # Append along sequence dimension (dim=2).
            self.K_cache = torch.cat([self.K_cache, K_new], dim=2)
            self.V_cache = torch.cat([self.V_cache, V_new], dim=2)

        return self.K_cache, self.V_cache

    def reset(self):
        """Clear the cache (e.g., before starting a new generation)."""
        self.K_cache = None
        self.V_cache = None


# --- Demo ---
if __name__ == "__main__":
    B, H, head_dim = 1, 4, 64
    cache = CacheKV()

    # Step 1: first token
    K1 = torch.randn(B, H, 1, head_dim)
    V1 = torch.randn(B, H, 1, head_dim)
    K, V = cache.update(K1, V1)
    print(f"After step 1: K shape = {K.shape}")  # (1, 4, 1, 64)

    # Step 2: second token
    K2 = torch.randn(B, H, 1, head_dim)
    V2 = torch.randn(B, H, 1, head_dim)
    K, V = cache.update(K2, V2)
    print(f"After step 2: K shape = {K.shape}")  # (1, 4, 2, 64)

    # Step 3: third token
    K3 = torch.randn(B, H, 1, head_dim)
    V3 = torch.randn(B, H, 1, head_dim)
    K, V = cache.update(K3, V3)
    print(f"After step 3: K shape = {K.shape}")  # (1, 4, 3, 64)

    # Reset for new generation
    cache.reset()
    print(f"\nAfter reset: K_cache = {cache.K_cache}")
