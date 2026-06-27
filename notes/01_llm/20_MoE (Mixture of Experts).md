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

## Load Balancing (L5 Critical)

### The Problem

Without load balancing, the router may collapse to using only a few experts:

```
Expert 0: 80% of tokens  ← overloaded
Expert 1: 15% of tokens
Expert 2: 3% of tokens   ← underutilized, never learns
Expert 3: 2% of tokens   ← underutilized, never learns
```

This creates a vicious cycle: popular experts get better → router sends more tokens to them → unpopular experts never improve.

### Auxiliary Load Balancing Loss

Add a loss term that penalizes uneven expert usage:

```
L_balance = α * N * Σ f_i * P_i
```

Where:
- `f_i` = fraction of tokens routed to expert `i`
- `P_i` = average router probability for expert `i`
- `N` = number of experts
- `α` = balancing weight (typically 0.01)

When all experts are equally used, `L_balance = α * N * (1/N) * (1/N) * N = α/N` (minimum). When one expert dominates, the loss increases.

### Capacity Factor

To prevent any single expert from being overwhelmed, MoE systems use a **capacity factor**:

```
capacity = (tokens_per_batch / N_experts) * capacity_factor
```

- `capacity_factor = 1.0`: each expert handles exactly its fair share.
- `capacity_factor = 1.25`: allows 25% overflow — some flexibility but bounded.
- Tokens exceeding capacity are **dropped** (or routed to next-best expert).

### Token Dropping

When an expert is at capacity, excess tokens are dropped:

```
if expert_load[expert_i] >= capacity:
    # Drop token or route to overflow expert
    token.dropped = True
```

- **Training**: dropped tokens don't contribute to loss (wasted compute).
- **Inference**: dropped tokens degrade output quality.
- **Goal**: minimize drops via good load balancing.

---

## DeepSeek MoE Innovations

### 1. Fine-Grained Expert Segmentation

Instead of 8 large experts, use many small experts:

```
Standard MoE:  8 experts, each with D_ff parameters
DeepSeek MoE:  64 experts, each with D_ff/8 parameters
              Route top-6 instead of top-2
```

**Benefits:**
- More expert specialization (each expert handles a narrower domain).
- More combinations: `C(64, 6)` >> `C(8, 2)` — more diverse routing.
- Same total parameters, same active compute, better quality.

### 2. Shared Experts

Some experts are **always active** (not routed):

```
Standard MoE:  route to top-2 of 8 experts
DeepSeek MoE:  2 shared experts (always active) + route to top-6 of 64 routed experts
```

**Benefits:**
- Shared experts handle **common knowledge** (syntax, basic reasoning).
- Routed experts handle **specialized knowledge** (domain-specific).
- Reduces redundancy — no need for every routed expert to learn common patterns.

### 3. Expert-Specific Learning Rate

Different experts get different learning rates based on their utilization:

```
LR_i = base_lr * (1 + α * (target_freq - actual_freq_i))
```

Underutilized experts get higher LR → catch up faster.

---

## MoE Models in Practice

| Model | Total Params | Active Params | Experts | Active | Architecture |
|---|---|---|---|---|---|
| Mixtral 8x7B | 47B | 13B | 8 | 2 | Standard Top-2 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | Fine-grained + shared |
| Grok-1 | 314B | ~86B | 8 | 2 | Standard Top-2 |
| GPT-4 (rumored) | ~1.8T | ~280B | 16 | 2 | Standard MoE |
| Switch Transformer | 1.6T | - | 2048 | 1 | Top-1 routing |

### Key Observations

- Mixtral 8x7B: 47B total but only 13B active → similar inference cost to 13B dense model, but with 47B capacity.
- DeepSeek-V2: 236B total, 21B active → 11x parameter-to-compute ratio.
- The ratio of total/active parameters is the key MoE metric.

---

## MoE Training Challenges

### 1. Routing Instability

The router can be unstable early in training — small changes in router weights cause large changes in routing patterns.

**Mitigation**: router z-loss (penalize large router logits), warmup with dense training.

### 2. Expert Underutilization

Some experts receive very few tokens → their weights don't update → they never improve.

**Mitigation**: load balancing loss, capacity factor, expert dropout.

### 3. Communication Overhead

In distributed training, experts are split across GPUs. Each token may need to be sent to a different GPU:

```
GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

Token routed to Expert 3 → send from GPU 0 to GPU 1 → compute → send back
```

This **all-to-all communication** is expensive and can dominate training time.

**Mitigation**: expert parallelism with locality-aware routing, overlapping communication with computation.

### 4. Memory vs Compute Mismatch

MoE models have high memory (all experts loaded) but low compute (only a few active). This is the opposite of dense models:

```
Dense 13B:  26 GB memory, 13B FLOPs/token
MoE 47B:   94 GB memory, 13B FLOPs/token
```

The MoE model needs 3.6x more memory for the same compute. This makes MoE **memory-bound** during inference.

---

## MoE Serving Considerations

### Expert Parallelism

Distribute experts across GPUs:

```
GPU 0: Expert 0, 1 (always loaded)
GPU 1: Expert 2, 3 (always loaded)
GPU 2: Expert 4, 5 (always loaded)
GPU 3: Expert 6, 7 (always loaded)

Each token: route to correct GPU → compute → gather results
```

### Latency Impact

- **Extra routing overhead**: router computation (small but non-zero).
- **All-to-all communication**: sending tokens to the right expert's GPU.
- **Load imbalance**: if one expert is hot, that GPU becomes a bottleneck.

### Throughput vs Latency

MoE is better for **throughput** (more capacity per FLOP) but can be worse for **latency** (communication overhead). For latency-critical serving, dense models may be preferred.

---

## L5 Interview Q&A

### Q: "When would you choose MoE over a dense model?"

**Choose MoE when:**
- You want more capacity without proportional compute increase.
- Inference throughput matters more than latency.
- You have enough memory to hold all experts.
- You can handle the routing and load balancing complexity.

**Choose dense when:**
- Latency is critical (MoE has communication overhead).
- Memory is constrained (MoE needs all experts loaded).
- Simplicity is important (MoE adds routing, load balancing, expert parallelism).
- Model size is small enough that dense is feasible.

### Q: "How does expert parallelism work?"

1. Partition experts across GPUs (e.g., 8 experts across 4 GPUs → 2 experts each).
2. All GPUs compute the router logits for all tokens.
3. Each GPU determines which tokens are routed to its experts.
4. **All-to-all communication**: tokens are sent to the GPU that holds their assigned expert.
5. Each GPU computes the expert FFN for its tokens.
6. **All-to-all communication**: results are sent back to the original GPU.
7. Combine expert outputs weighted by router probabilities.

The all-to-all communication is the main bottleneck — it's O(N * D) where N is tokens and D is hidden dimension.

### Q: "What's the memory breakdown of a MoE model?"

```
Mixtral 8x7B (47B total, 13B active):
  Attention layers: ~1.3B params (shared, not multiplied by 8)
  FFN layers (MoE): ~45B params (8 experts × ~5.6B each)
  Embeddings + output: ~0.5B params

  Memory (FP16): 47B * 2 = 94 GB
  Active per token: 13B * 2 = 26 GB worth of compute
```

The model needs 94 GB of memory even though only 26 GB worth of parameters are active per token. This is the fundamental MoE tradeoff: **memory for compute**.

### Q: "How does DeepSeek's fine-grained MoE compare to standard MoE?"

Standard MoE (Mixtral): 8 large experts, route top-2 → `C(8,2) = 28` possible expert combinations.

DeepSeek MoE: 64 small experts + 2 shared, route top-6 → `C(64,6) = 74,974,368` possible combinations.

The fine-grained approach has **3 million times more routing combinations**, enabling much more diverse token-expert assignments. Each expert specializes in a narrower domain, leading to better quality at the same compute budget.

### Q: "How do you handle load imbalance in MoE serving?"

1. **Capacity factor**: set a hard limit on tokens per expert.
2. **Dynamic batching**: adjust batch composition to balance expert load.
3. **Expert replication**: replicate hot experts across multiple GPUs.
4. **Overflow routing**: route excess tokens to the next-best expert.
5. **Monitoring**: track expert utilization in real-time, alert on imbalance.

### Q: "Why don't we route at the sequence level instead of token level?"

Sequence-level routing (all tokens in a sequence go to the same expert) reduces communication overhead but:
- **Less fine-grained**: different tokens in a sequence may need different experts.
- **Worse load balancing**: sequences vary in length, making balance harder.
- **Lower quality**: token-level routing allows specialization within a sequence.

Token-level routing is standard because it gives better quality and load balancing, despite higher communication cost.

---

## Common Traps

- Forgetting to renormalize top-k router probabilities.
- Using all experts (defeats MoE purpose).
- Ignoring load balancing (some experts collapse).
- Mixing token-level routing and sequence-level routing.
- Forgetting that MoE needs all experts in memory (not just active ones).
- Not handling token dropping gracefully (dropped tokens = quality loss).

---

## Interview Sound Bites

- MoE = conditional computation: scale parameters faster than per-token FLOPs.
- **Mixtral 8x7B**: 47B total, 13B active — same inference cost as 13B dense, 3.6x capacity.
- **Load balancing loss**: `α * N * Σ f_i * P_i` — prevents expert collapse.
- **Capacity factor**: hard limit on tokens per expert — excess tokens dropped.
- **DeepSeek MoE**: fine-grained experts (64 small vs 8 large) + shared experts (always active) → better specialization.
- **Expert parallelism**: experts split across GPUs, all-to-all communication is the bottleneck.
- MoE is **memory-bound** at inference: all experts loaded, only a few active.
- MoE better for throughput; dense better for latency.
- Token-level routing > sequence-level routing (finer specialization, better balance).
- Training challenges: routing instability, expert underutilization, communication overhead.
