# FlashAttention Intuition

## What Changes, What Stays Same

- **Time complexity:** still `O(N^2)`
- **Space complexity:** `O(N^2)` -> `O(N)` (practical memory reduction)

## Standard Attention Bottleneck

Standard attention materializes full score/weight matrices:

- `scores = QK^T` (N x N)
- `A = softmax(scores)` (N x N)

For long N, memory traffic to HBM becomes the bottleneck.

## FlashAttention Idea

Do attention in small blocks that fit on-chip SRAM.

- Load tile of Q/K/V
- Compute partial scores
- Apply online softmax stats (running max + running sum)
- Accumulate output
- Discard block

Never store full NxN matrix.

## Why Online Softmax Matters

Softmax needs normalization over a full row.

FlashAttention keeps row-wise running stats so blocks can be processed incrementally while staying exact (or very close depending on variant).

## Practical Outcome

- Much lower memory traffic
- Better throughput on long sequences
- Enables larger context windows

## What It Is Not

- Not a windowed-attention approximation by default
- Not changing model architecture
- Not reducing arithmetic complexity class

## Online Softmax Algorithm (The Key Trick)

### The Problem

Softmax requires the **full row** to normalize:

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

If we process the row in blocks, we don't know the denominator until we've seen all blocks.

### The Solution: Running Statistics

Maintain two running values as you scan blocks:

1. `m` = running max (for numerical stability)
2. `l` = running sum of `exp(x_i - m)`

When processing a new block `B`:

```
m_new = max(m_old, max(B))
l_new = l_old * exp(m_old - m_new) + Σ exp(B - m_new)
```

The `exp(m_old - m_new)` term **rescales** the old sum to the new max. This is the key insight — the old statistics are corrected, not discarded.

### Pseudocode

```python
def online_softmax_row(x_blocks):
    m = -inf  # running max
    l = 0     # running sum
    for block in x_blocks:
        m_new = max(m, max(block))
        l = l * exp(m - m_new) + sum(exp(block - m_new))
        m = m_new
    # l now equals sum(exp(x - max(x))) for the full row
    return l, m
```

### Why This Is Exact

This is not an approximation. The final `l` and `m` are exactly what you'd get if you computed softmax over the full row. The running correction ensures no information is lost.

---

## Tiling Explained

### GPU Memory Hierarchy

```
HBM (High Bandwidth Memory): ~40-80 GB, ~1-2 TB/s
SRAM (On-chip): ~192 KB per SM, ~19 TB/s
```

SRAM is ~10x faster than HBM but ~400x smaller. FlashAttention keeps tiles in SRAM and never writes intermediate results to HBM.

### Tiling Strategy

```
Q: [Q1 | Q2 | Q3 | ...]   ← row tiles (query blocks)
K: [K1 | K2 | K3 | ...]   ← column tiles (key blocks)
V: [V1 | V2 | V3 | ...]

For each Qi:
    Initialize: Oi = 0, mi = -inf, li = 0
    For each Kj, Vj:
        Sij = Qi @ Kj^T           # partial scores (in SRAM)
        mij = max(Sij)             # block max
        Pij = exp(Sij - mij)       # block softmax (unnormalized)
        li = li * exp(mi - mij) + sum(Pij)   # update running stats
        Oi = Oi * exp(mi - mij) / ... + Pij @ Vj  # accumulate output
        mi = max(mi, mij)
    Oi = Oi / li                   # final normalization
```

### Tile Size Selection

- Too large: doesn't fit in SRAM.
- Too small: too many iterations, overhead dominates.
- Typical: `Br` (query block) × `Bc` (key block) chosen to fill SRAM.
- FlashAttention-2 uses `Br = 64, Bc = 64` typically.

---

## FlashAttention-2 vs FlashAttention-1

| | FA1 | FA2 |
|---|---|---|
| Parallelization | Over sequence length (N) | Over batch × heads × N (more parallel) |
| Non-MATM FLOPs | Higher (rescaling operations) | Minimized |
| Backward pass | Recomputes with same tiling | Optimized recompute, fewer HBM reads |
| Speedup vs standard | ~2-4x | ~5-8x |
| Key insight | IO-aware tiling | Better work partitioning + less overhead |

### FlashAttention-3 (2024)

FA3 adds:
- **Asynchronous computation**: overlap softmax with GEMM (Tensor Cores).
- **FP8 support**: lower precision for even higher throughput.
- **Warp-specialization**: different warp groups handle different tasks.
- ~1.5-2x faster than FA2 on H100 GPUs.

---

## Memory HBM Read/Write Analysis

### Standard Attention

```
1. Read Q, K from HBM          → O(N * d)
2. Compute S = QK^T, write S   → O(N²) writes
3. Read S, compute P = softmax(S), write P → O(N²) reads + writes
4. Read P, V, compute O = PV   → O(N² + N*d) reads
5. Write O                      → O(N * d) writes

Total HBM traffic: O(N² + N*d) ≈ O(N²) for large N
```

### FlashAttention

```
1. Read Q, K, V in tiles       → O(N * d) per tile
2. Compute in SRAM (no HBM)    → 0 HBM for intermediates
3. Write O in tiles             → O(N * d)

Total HBM traffic: O(N * d * N / Br) = O(N² * d / Br)
```

Since `Br` (block size) is constant, this is still O(N²) in big-O, but the **constant factor** is much smaller — `d/Br` instead of 1. For `d=128, Br=64`, that's a 2x reduction. Combined with avoiding intermediate writes, the practical speedup is 5-8x.

### The Real Win

The speedup isn't from reducing FLOPs — it's from reducing **HBM bandwidth usage**. GPUs are often memory-bandwidth bound, not compute bound. FlashAttention shifts the bottleneck from memory to compute, which is what GPUs are designed for.

---

## L5 Interview Q&A

### Q: "Is FlashAttention an approximation?"

**No.** FlashAttention computes the exact same result as standard attention. The online softmax trick with running max/sum corrections is mathematically exact. There is zero quality loss.

### Q: "Why is FlashAttention faster if it has the same FLOPs?"

GPUs are often **memory-bandwidth bound**, not compute bound. Standard attention writes the N×N matrix to HBM three times (scores, softmax weights, output). FlashAttention keeps everything in SRAM and only reads/writes the final output. The speedup comes from reduced HBM traffic, not reduced computation.

### Q: "How does FlashAttention handle the backward pass?"

During backward, FlashAttention **recomputes** the attention statistics (S, P) from Q, K, V in tiles, rather than storing them from the forward pass. This trades extra compute for reduced memory — and since GPUs are memory-bound, this is faster.

### Q: "What are the limitations of FlashAttention?"

1. **Implementation complexity**: requires custom CUDA kernels — can't be done with pure PyTorch.
2. **Fixed sequence length**: some implementations require padded sequences for efficient tiling.
3. **Small sequences**: for very short sequences (< 512), the overhead of tiling may not be worth it.
4. **Custom attention variants**: modifications to attention (e.g., additive biases like ALiBi) require kernel modifications.

### Q: "How does FlashAttention interact with KV cache?"

During decoding (single token), the attention matrix is `(1, N)` — no N×N materialization. FlashAttention's main benefit is during **prefill** (processing the prompt) where the full N×N matrix would be materialized. During decode, standard attention is already efficient.

### Q: "When would you NOT use FlashAttention?"

- **Very short sequences** (< 256): overhead may exceed benefit.
- **Custom attention patterns** that can't be tiled (e.g., some sparse attention variants).
- **Non-GPU hardware** where the memory hierarchy is different.
- **When you need the attention weights** for interpretability — FlashAttention doesn't materialize them.

---

## Interview Sound Bites

- FlashAttention is an **exact** (not approximate) IO-aware attention algorithm: same O(N²) compute, O(N) space, 5-8x faster wall-clock.
- The key trick is **online softmax**: maintain running max and running sum, correct as you scan blocks — mathematically exact.
- Tiling keeps computation in fast SRAM, never writes the N×N matrix to slow HBM.
- The speedup comes from reduced **memory bandwidth**, not reduced FLOPs — GPUs are memory-bound.
- FlashAttention-2 improves parallelization and reduces overhead; FA3 adds async computation and FP8.
- Backward pass recomputes attention from Q/K/V rather than storing intermediates — trades compute for memory.
- Main benefit is during **prefill**; decode with KV cache is already efficient.
- Not an approximation — zero quality loss vs standard attention.
