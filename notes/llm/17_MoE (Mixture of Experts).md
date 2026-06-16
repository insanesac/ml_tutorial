# MoE (Mixture of Experts)

## What Problem Is MoE Solving?

In a standard Transformer FFN:

```python
hidden = relu(X @ W1 + b1)
output = hidden @ W2 + b2
```

Every token goes through the same FFN.

This is simple, but scaling it means more compute for every token.

MoE solves this by increasing model capacity without activating all parameters per token.

---

## Core Idea

Instead of one FFN, have many FFNs (experts):

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- ...

Each expert is just a normal FFN.

A router decides which experts a token should use.

---

## Why Not Run All Experts?

Naive approach:

```
token
  ↓
E1 E2 E3 E4
```

This explodes compute.

MoE is useful only when routing is sparse (Top-1 / Top-2 / Top-k), so only a few experts run per token.

---

## Router

Router is a small linear layer:

```python
router_logits = X @ W_router
```

If `num_experts = 4`, then:

- `router_logits.shape = (B, N, 4)`

Example logits for one token:

`[2.0, 0.5, 4.0, 1.0]`

After softmax:

`[0.11, 0.02, 0.81, 0.06]`

Interpretation:
- Best expert: index `2`
- Second best: index `0`

---

## Top-1 MoE

Use only the best expert:

```python
expert_idx = np.argmax(router_probs)
```

Token runs through one expert only.

---

## Top-2 MoE

Common in modern models.

Pick top-2 experts and combine outputs by router weights.

Find top-2 indices in NumPy:

```python
top2 = np.argsort(probs)[::-1][:2]
```

For `probs = [0.11, 0.02, 0.81, 0.06]`:

- `top2 = [2, 0]`

---

## MoE Forward Pass (Conceptual)

1. Compute router probabilities
2. Pick top-k expert indices
3. Run selected experts only
4. Weighted combine their outputs

---

## Interview-Style Implementation

```python
def moe_forward(x, experts, W_router, k=2):
    # x shape could be (D,) for single token or (B, N, D) for batched tokens.
    router_logits = x @ W_router
    router_probs = softmax(router_logits)

    topk_idx = np.argsort(router_probs)[::-1][:k]

    topk_probs = router_probs[topk_idx]
    topk_probs = topk_probs / np.sum(topk_probs)

    output = 0
    for p, idx in zip(topk_probs, topk_idx):
        output += p * experts[idx](x)

    return output
```

---

## Why MoE Matters

Without MoE:

- 1 FFN with 100M params
- all 100M active per token

With MoE (example):

- 8 experts × 100M = 800M total params
- Top-2 routing activates only 2 experts

Result:

- huge model capacity
- much lower active compute per token than dense equivalent

This is why models like Mixtral and DeepSeek-style architectures use MoE.

---

## Key Interview Insight

MoE gives **conditional computation**:

- Dense model: every parameter used for every token
- MoE model: only selected experts used per token

So you scale parameters faster than per-token FLOPs.

---

## Common Traps

- Forgetting to renormalize top-k router probabilities
- Using all experts (defeats MoE purpose)
- Ignoring load balancing (some experts collapse)
- Mixing token-level routing and sequence-level routing

---

## One-Liner

MoE increases model capacity by routing each token to a small subset of experts (Top-k), enabling sparse activation and much better parameter-to-compute efficiency.
