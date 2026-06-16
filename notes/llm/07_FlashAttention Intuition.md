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

## Interview Sound Bite

FlashAttention is an IO-aware exact attention algorithm: same `O(N^2)` compute, much lower memory movement, so it runs faster in practice on GPUs.
