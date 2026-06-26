"""KV Cache for autoregressive decoding in NumPy.

During autoregressive generation, each new token only needs:
- Q for the new token (computed fresh each step)
- K and V for ALL tokens (including all previous ones)

Without cache: recompute K, V for all previous tokens every step → O(N^2) total.
With cache:    store previous K, V, only compute K, V for the new token → O(N) total.

Cache shape: (B, H, T, head_dim) where T grows by 1 each step.
New K/V shape: (B, H, 1, head_dim) — just the latest token.
"""

import numpy as np


def update_cache(cache, new_kv):
    """Append new key/value vectors to the KV cache.

    Args:
        cache:   existing KV cache, shape (B, H, T, head_dim) or None for first step.
        new_kv:  new key/value vectors, shape (B, H, 1, head_dim).

    Returns:
        Updated cache, shape (B, H, T+1, head_dim).
    """
    if cache is None:
        return new_kv
    return np.concatenate([cache, new_kv], axis=2)


# --- Demo ---
if __name__ == "__main__":
    B, H, head_dim = 1, 4, 64

    # Step 1: first token
    K_new = np.random.randn(B, H, 1, head_dim)
    V_new = np.random.randn(B, H, 1, head_dim)

    K_cache = update_cache(None, K_new)
    V_cache = update_cache(None, V_new)
    print(f"After step 1: K_cache shape = {K_cache.shape}")  # (1, 4, 1, 64)

    # Step 2: second token
    K_new = np.random.randn(B, H, 1, head_dim)
    V_new = np.random.randn(B, H, 1, head_dim)

    K_cache = update_cache(K_cache, K_new)
    V_cache = update_cache(V_cache, V_new)
    print(f"After step 2: K_cache shape = {K_cache.shape}")  # (1, 4, 2, 64)

    # Step 3: third token
    K_new = np.random.randn(B, H, 1, head_dim)
    V_new = np.random.randn(B, H, 1, head_dim)

    K_cache = update_cache(K_cache, K_new)
    V_cache = update_cache(V_cache, V_new)
    print(f"After step 3: K_cache shape = {K_cache.shape}")  # (1, 4, 3, 64)
