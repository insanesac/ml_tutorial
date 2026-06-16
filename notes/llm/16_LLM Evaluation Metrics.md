# LLM Evaluation Metrics

## Why Evaluation Is Hard for LLMs

Unlike classification, LLM outputs are open-ended text.

There is often no single correct answer.

Different metrics capture different aspects of quality.

---

## Perplexity

Already covered in `Perplexity, LayerNorm & RMSNorm.md`.

```
PPL = exp(Cross Entropy)
```

- Measures how well the model predicts a held-out test set.
- Lower = better.
- **Only works when you have ground-truth token sequences.**
- Cannot evaluate instruction-following or factual accuracy.

---

## BLEU (Bilingual Evaluation Understudy)

### What It Measures

Overlap between generated text and reference text using n-gram precision.

### Formula (simplified)

```
BLEU = BP * exp(Σ w_n * log(p_n))
```

Where:
- `p_n` = precision of n-grams (1-gram, 2-gram, 3-gram, 4-gram)
- `BP` = brevity penalty (penalizes short outputs)
- `w_n` = weights (usually uniform = 0.25 each)

### Intuition

```
Reference: "The cat sat on the mat"
Generated: "The cat sat on the mat"   -> high BLEU
Generated: "A cat is on a mat"        -> lower BLEU
Generated: "cat mat"                  -> very low BLEU (brevity penalty)
```

### Weaknesses

- Penalizes valid paraphrases.
- Does not capture meaning.
- Saturates — high BLEU does not mean high quality.

### Used For

Machine translation, summarization benchmarks.

---

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

### What It Measures

Overlap between generated and reference text, focused on **recall** (did you cover the reference?).

### Variants

| Variant | What It Measures |
|---|---|
| ROUGE-1 | Unigram overlap |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest Common Subsequence |

### ROUGE-L Intuition

Finds the longest sequence of tokens that appear in both generated and reference text in order (not necessarily contiguous).

### Used For

Summarization evaluation.

### Weaknesses

Same as BLEU — surface overlap, not semantic meaning.

---

## BLEU vs ROUGE

| | BLEU | ROUGE |
|---|---|---|
| Focus | Precision | Recall |
| Use case | Translation | Summarization |
| Reference needed | Yes | Yes |

---

## MMLU (Massive Multitask Language Understanding)

### What It Measures

Multiple-choice questions across 57 academic subjects.

```
Q: What is the powerhouse of the cell?
A) Nucleus  B) Mitochondria  C) Ribosome  D) Golgi

Correct: B
```

Score = fraction of questions answered correctly.

### Why It Matters

- Tests broad world knowledge.
- Easy to evaluate automatically.
- Used to compare models at scale.

### Limitation

- Multiple choice does not test generation quality.
- Susceptible to format hacking (models that guess well).

---

## Human Evaluation

The gold standard.

### Methods

- **Side-by-side** — show two responses, rater picks better one.
- **Likert scale** — rate response 1–5 on helpfulness, harmlessness, etc.
- **Elo ranking** — tournament-style, compute model Elo from pairwise preferences.

### Chatbot Arena

Crowdsourced human preference evaluation. Models are ranked by Elo.

Considered the most reliable real-world benchmark.

### Weaknesses

- Expensive.
- Slow.
- Rater disagreement.
- Positional bias (raters prefer first response).

---

## LLM-as-Judge

Use a strong model (e.g. GPT-4) to evaluate outputs.

```
Prompt: "Rate this response on helpfulness 1-10 and explain."
```

Scales better than human eval.

Risk: judges have their own biases and prefer outputs that look like their own style.

---

## Metric Selection Guide

| Task | Primary Metric |
|---|---|
| Language modeling | Perplexity |
| Machine translation | BLEU |
| Summarization | ROUGE-L |
| Knowledge / QA | MMLU, accuracy |
| Instruction following | Human eval / Chatbot Arena |
| Safety / alignment | Human eval, red-teaming |

---

## Interview Sound Bites

- Perplexity measures next-token prediction quality but not instruction following.
- BLEU measures precision of n-gram overlap; ROUGE measures recall.
- Both BLEU and ROUGE are surface-level — they miss semantic equivalence.
- MMLU tests broad knowledge via multiple choice at scale.
- Human eval is the gold standard but expensive; LLM-as-judge scales it.
- For production, track multiple metrics — no single metric captures everything.
