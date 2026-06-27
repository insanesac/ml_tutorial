# Transformer Family: Encoder, Decoder, BERT, GPT, T5

## Original Transformer

The original Transformer was built for machine translation:

```
English sentence → Encoder → Encoder representations → Decoder (cross-attention) → French sentence
```

---

## Encoder

**Purpose:** Read → Understand → Represent

- Uses **bidirectional self-attention** — every token can attend to every other token
- Used when we want **understanding**
- Examples: classification, NER, question answering, embeddings, retrieval

---

## Decoder

**Purpose:** Generate one token at a time

- Uses **masked self-attention** — each token can only attend to previous tokens
- Used when we want **generation**

---

## Cross Attention

Cross attention connects the decoder to the encoder:

```
Q = Decoder
K = Encoder
V = Encoder
```

Without cross-attention, the decoder has no access to the input sentence. It may generate fluent text, but it cannot translate or answer based on the source input.

---

## BERT (Encoder-only)

```
Input → Encoder (bidirectional self-attention) → Contextual representations
```

**Best for:** understanding

**Training:** Masked Language Modeling (MLM) — predict masked tokens using left and right context

**Cannot** naturally autoregressively generate text because it depends on future context.

---

## GPT (Decoder-only)

```
Prompt → Decoder (masked self-attention) → Next token
```

**Best for:** generation

**Training:** Next-token prediction

**Can** generate because it only depends on past tokens.

---

## T5 (Encoder + Decoder)

```
Input text → Encoder → Cross-attention → Decoder → Output text
```

T5 turns every NLP task into **text-to-text**:

| Task | Input | Output |
|---|---|---|
| Translation | `translate English to French: I love AI` | `J'aime l'IA` |
| Summarization | `summarize: long article` | `short summary` |
| Sentiment | `sentiment: I loved this movie` | `positive` |

---

## Simple Mental Model

```
Original Transformer = Encoder + Decoder
BERT  = Keep encoder, remove decoder
GPT   = Keep decoder, remove encoder and cross-attention
T5    = Keep encoder and decoder
```

---

## Attention Types Summary

| Attention Type | Used In | Meaning |
|---|---|---|
| Bidirectional self-attention | Encoder / BERT | Tokens see full input |
| Masked self-attention | Decoder / GPT | Tokens see only past |
| Cross-attention | Encoder-decoder / T5 | Decoder attends to encoder output |

---

## Interview Takeaway

BERT, GPT, and T5 are not totally separate ideas. They are different combinations of the same Transformer blocks:

- **Encoder** = understanding
- **Decoder** = generation
- **Cross-attention** = conditioning generation on input
