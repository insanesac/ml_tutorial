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
| Reasoning | GSM8K, MATH, AGIEval |
| Code generation | HumanEval, MBPP |
| Multi-turn dialogue | MT-Bench |

---

## Modern Benchmarks

### Knowledge & Reasoning

| Benchmark | What It Tests | Format |
|---|---|---|
| MMLU | 57 subjects, broad knowledge | Multiple choice |
| MMLU-Pro | Harder MMLU, 10 options | Multiple choice |
| AGIEval | SAT, LSAT, GRE-level reasoning | Multiple choice + generation |
| ARC | Grade-school science reasoning | Multiple choice |
| HellaSwag | Commonsense NLI | Multiple choice |
| Winogrande | Coreference resolution | Multiple choice |

### Math & Code

| Benchmark | What It Tests | Format |
|---|---|---|
| GSM8K | Grade-school math word problems | Exact match |
| MATH | Competition mathematics | Exact match |
| HumanEval | Python function generation | Pass@1 |
| MBPP | Basic Python programming | Pass@1 |
| LiveCodeBench | Competitive programming | Pass@1 |

### Instruction Following & Chat

| Benchmark | What It Tests | Format |
|---|---|---|
| MT-Bench | Multi-turn conversation quality | LLM judge (GPT-4) |
| AlpacaEval | Instruction following | LLM judge |
| Chatbot Arena | Human preference, Elo rating | Human pairwise |
| IFEval | Instruction following (verifiable) | Rule-based |

### Safety & Alignment

| Benchmark | What It Tests | Format |
|---|---|---|
| TruthfulQA | Hallucination / truthfulness | Multiple choice + generation |
| ToxiGen | Toxicity detection | Classification |
| BBQ | Bias evaluation | Multiple choice |
| AdvBench | Adversarial prompt robustness | Generation + safety check |

---

## Data Contamination (L5 Critical)

### The Problem

If benchmark data leaks into training data, the model "memorizes" answers rather than genuinely solving them:

```
Training data: includes GSM8K questions and answers
Evaluation: model scores 95% on GSM8K
Reality: model memorized, doesn't actually reason → fails on new problems
```

### Detection Methods

1. **String matching**: check if benchmark questions appear in training data.
2. **Perplexity comparison**: if model has abnormally low perplexity on benchmark questions vs similar questions, likely contaminated.
3. **Canary tokens**: insert unique tokens in benchmarks, check if model generates them.
4. **Rephrased evaluation**: rephrase benchmark questions — if performance drops significantly, model was memorizing.

### Mitigations

1. **Hold out benchmarks**: never include benchmark data in training.
2. **Dynamic benchmarks**: generate new questions (e.g., LiveCodeBench uses recent LeetCode problems).
3. **Contamination reporting**: papers should report contamination checks.
4. **Private eval sets**: maintain internal benchmarks not publicly available.

### Why This Matters for L5

- Many model benchmarks are **inflated** due to contamination.
- A model that scores 90% on MMLU might score 70% on a contamination-free version.
- Always question whether benchmark improvements are real or contamination-driven.

---

## LLM-as-Judge Best Practices

### Bias Mitigation

| Bias | Description | Mitigation |
|---|---|---|
| Position bias | Prefers first/last response | Randomize order, average over both |
| Length bias | Prefers longer responses | Length-normalized scoring |
| Self-preference | Prefers outputs from same model | Use different judge model |
| Verbosity bias | Prefers verbose, detailed responses | Explicit scoring rubric |

### G-Eval (Chain-of-Thought Judging)

```
1. Define evaluation criteria (e.g., "score 1-5 on coherence")
2. Ask judge LLM to first explain its reasoning (CoT)
3. Then provide the score
4. Use the score's probability (not just argmax) for finer granularity
```

```python
def g_eval(judge_model, text, criteria):
    prompt = f"""
    You will evaluate a text on: {criteria}
    
    Text: {text}
    
    First, explain your reasoning step by step.
    Then, provide a score from 1 to 5.
    """
    response = judge_model.generate(prompt)
    return parse_score(response)
```

### Pairwise vs Pointwise

| | Pairwise | Pointwise |
|---|---|---|
| Method | Compare A vs B | Score each independently |
| Sensitivity | Higher (relative) | Lower (absolute) |
| Cost | 1 call per pair | 1 call per item |
| Best for | Ranking models | Absolute quality |

---

## Calibration

### What It Is

A model is **calibrated** if its confidence matches its accuracy:

```
Calibrated:   90% confidence → 90% correct
Uncalibrated: 90% confidence → 60% correct (overconfident)
```

### Why It Matters

- LLMs are typically **overconfident** — they state falsehoods with high confidence.
- Calibration affects trust — users need to know when to trust the model.
- Important for safety-critical applications (medical, legal).

### Measurement

**Expected Calibration Error (ECE)**:

```
1. Bin predictions by confidence (e.g., 0-10%, 10-20%, ..., 90-100%)
2. For each bin, compute |confidence - accuracy|
3. ECE = weighted average across bins
```

Lower ECE = better calibration.

### Improving Calibration

1. **Temperature scaling**: increase temperature to soften confident predictions.
2. **Label smoothing**: during training, prevents overconfidence.
3. **Verbalized confidence**: ask the model to state its confidence level.
4. **Sampling**: sample multiple responses and check consistency.

---

## L5 Interview Q&A

### Q: "How would you evaluate a new LLM for production deployment?"

1. **Automated benchmarks**: MMLU, GSM8K, HumanEval, IFEval — fast, reproducible.
2. **LLM-as-judge**: MT-Bench, AlpacaEval — scalable quality assessment.
3. **Human eval**: Chatbot Arena style pairwise comparison — gold standard.
4. **Domain-specific eval**: custom eval set for your use case (e.g., medical QA).
5. **Safety eval**: TruthfulQA, red-teaming, toxicity checks.
6. **Contamination check**: verify benchmarks aren't in training data.
7. **Calibration**: measure ECE to ensure confidence matches accuracy.
8. **Latency/cost**: measure TTFT, TPOT, tokens/sec at target batch size.

### Q: "Why is BLEU not suitable for evaluating LLMs?"

1. **Surface-level**: BLEU measures n-gram overlap, not meaning. "The cat sat on the mat" and "A feline rested on the rug" have zero BLEU overlap but identical meaning.
2. **Designed for MT**: BLEU was created for machine translation where output should match reference closely. LLMs generate diverse, creative outputs.
3. **Penalizes valid alternatives**: BLEU penalizes correct responses that use different words than the reference.
4. **No instruction following**: BLEU can't measure whether the model followed instructions (format, style, constraints).

### Q: "What's the difference between Chatbot Arena and MMLU?"

| | MMLU | Chatbot Arena |
|---|---|---|
| Type | Static benchmark | Live human evaluation |
| Format | Multiple choice | Open-ended generation |
| What it measures | Knowledge | Human preference |
| Contamination risk | High | Low (dynamic) |
| Cost | Free (automated) | Expensive (human) |
| Coverage | 57 subjects | Open-ended |

MMLU tests knowledge; Chatbot Arena tests real-world preference. Both are needed — they measure different things.

### Q: "How do you detect if your model is hallucinating in production?"

1. **RAG grounding**: check if generated claims are supported by retrieved context (RAGAS faithfulness).
2. **Self-consistency**: sample multiple responses, check if they agree.
3. **Confidence estimation**: use verbalized confidence or token probabilities.
4. **Fact-checking pipeline**: cross-reference with knowledge base.
5. **Monitoring**: track user feedback (thumbs down), flag low-confidence outputs.
6. **Citation requirement**: force the model to cite sources — if it can't, it's likely hallucinating.

### Q: "What metrics would you track for an LLM serving system?"

**Quality metrics:**
- Faithfulness (RAGAS), answer relevance, hallucination rate
- User feedback (thumbs up/down, ratings)
- Task completion rate (did the user's goal get met?)

**Performance metrics:**
- TTFT (time to first token), TPOT (time per output token)
- Throughput (tokens/sec), latency percentiles (p50, p95, p99)
- Error rate, timeout rate

**Safety metrics:**
- Safety classifier flag rate
- Red-team success rate
- Jailbreak detection rate

**Business metrics:**
- Cost per query, cost per token
- User retention, session length
- Conversion rate (if applicable)

---

## Interview Sound Bites

- Perplexity measures next-token prediction quality but not instruction following or reasoning.
- BLEU measures precision of n-gram overlap; ROUGE measures recall — both are surface-level.
- MMLU tests broad knowledge via multiple choice — high contamination risk.
- **Chatbot Arena** (Elo rating) is the gold standard for human preference — dynamic, hard to contaminate.
- **LLM-as-judge** scales human eval but has biases (position, length, self-preference) — mitigate with randomization and rubrics.
- **G-Eval**: chain-of-thought judging — ask judge to reason before scoring.
- **Data contamination**: benchmarks leaking into training data inflates scores — detect with perplexity checks and canary tokens.
- **Calibration**: LLMs are typically overconfident — measure ECE, improve with temperature scaling.
- **No single metric captures everything** — track multiple metrics across quality, performance, safety, and business dimensions.
- For production: automated benchmarks (fast) + LLM-as-judge (scalable) + human eval (gold standard) + domain-specific eval.
