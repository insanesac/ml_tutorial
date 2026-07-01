# Beam Search

## Motivation

Greedy decoding always chooses the highest probability token at every step. This local decision may produce a **globally suboptimal** sequence.

Beam Search explores multiple candidate sequences simultaneously.

## Greedy Decoding

```
Step 1: Choose Best Token → Step 2: Choose Best Token → Final Sequence
```

- Fast
- May miss better overall sequences

## Beam Search

Maintain **K** best partial sequences (beam width).

```
Beam Width = 3

Start
  ↓
Candidate A | Candidate B | Candidate C
  ↓ Expand all
  ↓ Keep Best 3
  ↓ Expand Again
  ↓ Repeat
```

## Beam Score

Each sequence accumulates log probabilities:

```
Score = log(p₁) + log(p₂) + ... + log(pₙ)
```

Highest score wins.

## Length Normalization

Longer sequences naturally receive lower log probabilities. Normalize scores by sequence length to avoid bias toward shorter outputs.

## Complexity

| | Complexity |
|---|---|
| Greedy | O(V) |
| Beam Search | O(K × V) |

Where V = vocabulary size, K = beam width

## Advantages

- Better global sequences
- Improved translation quality
- Better summarization

## Disadvantages

- Slower
- Higher memory
- Less diverse outputs
- Still approximate

## Why Modern Chat LLMs Rarely Use Beam Search

Beam Search tends to produce:
- Generic responses
- Less creativity

Modern chat models prefer **temperature sampling**, **top-k**, **top-p** for more natural conversations.

Beam Search remains common in:
- Machine Translation
- Speech Recognition
- OCR
- Sequence Generation

## Interview Summary

Beam Search is a decoding algorithm that maintains multiple candidate sequences during generation instead of greedily selecting the highest probability token at each step. By exploring several hypotheses simultaneously, it often produces better overall sequences at the cost of increased computation.
