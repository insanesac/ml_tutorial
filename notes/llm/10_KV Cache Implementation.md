# KV Cache Implementation

## Goal

During decoding, reuse previous keys/values so each step computes attention only for the new token.

## Without KV Cache

At time `t`, recompute K/V for all `t` tokens.

Per-step work grows with sequence length.

## With KV Cache

Store previous K/V and append only new K/V.

Per-step computation becomes much cheaper.

## Tensor Shapes

For one layer:

- `K_cache`: `(B, H_kv, T, d)`
- `V_cache`: `(B, H_kv, T, d)`

At step `t`, append `k_t`, `v_t` with shape `(B, H_kv, 1, d)`.

## Minimal Pseudocode

```python
def decode_step(x_t, cache, Wq, Wk, Wv):
    # x_t: (B, 1, D)
    q_t = project_q(x_t, Wq)                # (B, H, 1, d)
    k_t = project_k(x_t, Wk)                # (B, H_kv, 1, d)
    v_t = project_v(x_t, Wv)                # (B, H_kv, 1, d)

    # append to cache
    cache['K'] = np.concatenate([cache['K'], k_t], axis=2)
    cache['V'] = np.concatenate([cache['V'], v_t], axis=2)

    # attention of current query against all cached keys
    out_t = attention(q_t, cache['K'], cache['V'])
    return out_t, cache
```

## Causal Mask in Decoding

For one-token decode, causal masking is implicit because query length is 1 and keys only include past+current tokens.

## Common Engineering Notes

- Pre-allocate cache for max length to avoid repeated concat
- Keep cache in contiguous GPU memory
- Quantize KV cache (fp8/int8 variants) for memory reduction

## Complexity Intuition

- No cache: repetitive recomputation of past K/V
- With cache: compute new K/V once, reuse past

## Interview Sound Bites

- KV cache stores K/V, not Q.
- New token gets new Q, so attention scores are recomputed each step.
- Cache is the core reason modern autoregressive serving is fast enough.
