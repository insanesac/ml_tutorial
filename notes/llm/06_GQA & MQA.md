# GQA & MQA (Grouped / Multi-Query Attention)

## Motivation

In autoregressive decoding, KV cache dominates memory.

Standard MHA stores separate K/V for every head:

`KV cache ~ O(H * N * D_head)`

GQA and MQA reduce KV memory.

## Standard MHA

- Query heads: `H`
- Key heads: `H`
- Value heads: `H`

Highest quality, highest KV memory.

## MQA (Multi-Query Attention)

- Query heads: `H`
- Key heads: `1`
- Value heads: `1`

All query heads share same K and V.

### Pros
- Big KV cache reduction
- Faster decoding

### Cons
- Can hurt quality vs full MHA

## GQA (Grouped-Query Attention)

Middle ground between MHA and MQA.

- Query heads: `H`
- Key/Value heads: `G` where `1 < G < H`

Each group of query heads shares one K/V head.

## KV Memory Comparison

For fixed `N, D_head`:

- MHA: proportional to `H`
- GQA: proportional to `G`
- MQA: proportional to `1`

So `MQA < GQA < MHA` in memory.

## Why GQA Is Popular

It gives most of MQA’s speed/memory gain with less quality drop.

This is why many modern serving stacks prefer GQA.

## Implementation Intuition

1. Compute Q with `H` heads as usual.
2. Compute K/V with `G` heads.
3. Map each query head to its KV group.
4. Attention per query head uses grouped K/V.

## Interview Sound Bites

- GQA/MQA are decoding-time optimizations via smaller KV cache.
- They do not reduce Q heads; they reduce K/V heads.
- MQA = extreme sharing (1 KV head), GQA = partial sharing.
- Tradeoff is quality vs latency/memory.
