# Scaling Laws

## What Are Scaling Laws?

Empirical relationships between model performance and three variables:

- **N** — number of parameters
- **D** — number of training tokens
- **C** — compute budget (FLOPs)

The key finding: loss follows predictable power laws as you scale any of these.

---

## OpenAI Scaling Laws (Kaplan et al., 2020)

Loss decreases as a power law with scale:

```
L(N) ∝ N^(-α)
L(D) ∝ D^(-β)
```

Key finding: **model size matters more than dataset size** for a fixed compute budget.

Recommendation: train large models on less data.

---

## Chinchilla Scaling Laws (Hoffmann et al., 2022)

Revisited Kaplan's findings with better experiments.

### Core Result

For a given compute budget C, the optimal strategy is:

```
N_optimal ∝ C^0.5
D_optimal ∝ C^0.5
```

**Parameters and tokens should scale equally.**

### Chinchilla Rule of Thumb

```
Optimal tokens ≈ 20 × N
```

To train a model with N parameters optimally, use ~20N training tokens.

### Examples

| Model | Params | Tokens Used | Chinchilla Optimal Tokens |
|---|---|---|---|
| GPT-3 | 175B | 300B | ~3.5T (undertrained) |
| Chinchilla | 70B | 1.4T | ~1.4T (optimal) |
| LLaMA 3 | 8B | 15T | (overtrained intentionally) |

### Why Overtrain Intentionally?

Chinchilla optimal = best loss for a given compute budget during training.

But for **inference**, a smaller model trained on more data is cheaper to serve.

LLaMA overtrains small models so they are inference-efficient.

---

## The Three Axes

| Axis | Effect | Bottleneck |
|---|---|---|
| More parameters (N) | Better capacity | Memory |
| More tokens (D) | Better generalization | Data collection |
| More compute (C) | Better overall | Cost |

---

## Emergent Abilities

At certain scale thresholds, capabilities appear suddenly:

- Few-shot learning
- Chain-of-thought reasoning
- Code generation

Not predictable from small-scale extrapolation.

Still not fully understood.

---

## Practical Implications for L5

| Question | Answer |
|---|---|
| Why use a 7B model over 70B for inference? | Cheaper, Chinchilla-overtrained small models can match larger undertrained ones |
| Why does GPT-3 underperform vs smaller newer models? | It was undertrained relative to its parameter count |
| How do you decide training token budget? | Chinchilla: ~20× parameter count |
| What limits scale in practice? | Compute cost, memory, data quality |

---

## Scaling Law Formulas

### Kaplan (OpenAI, 2020)

```
L(N) = (N_c / N)^α_N           (loss vs parameters)
L(D) = (D_c / D)^α_D           (loss vs data)
L(C) = (C_c / C)^α_C           (loss vs compute)
```

Where:
- `L` = loss (cross entropy)
- `N` = parameter count
- `D` = token count
- `C` = compute (FLOPs)
- Exponents: `α_N ≈ 0.076`, `α_D ≈ 0.095`, `α_C ≈ 0.05`

### Chinchilla (DeepMind, 2022)

```
L(N, D) = E + A/N^α + B/D^β
```

Where:
- `E` = irreducible loss (entropy of natural language)
- `A/N^α` = loss from finite model capacity
- `B/D^β` = loss from finite data
- Optimal: `N ∝ D` (scale parameters and data equally)

### Compute-Optimal Allocation

Chinchilla showed that for a fixed compute budget `C`:

```
N_optimal ∝ C^0.5
D_optimal ∝ C^0.5
```

You should scale parameters and data **equally** — not put most compute into a bigger model with less data (which is what GPT-3 did).

### FLOPs Estimation

```
C ≈ 6 * N * D    (forward + backward FLOPs)
```

Where `N` = parameters, `D` = tokens. This is the standard approximation for transformer training cost.

---

## Overtraining Economics (L5 Critical)

### Why Modern Small Models Are Overtrained

**Chinchilla-optimal** training minimizes loss for a given compute budget. But **inference cost** changes the calculus:

```
Chinchilla-optimal 7B: train on 140B tokens → loss L1
Overtrained 7B:       train on 2T tokens   → loss L2 < L1 (better!)
```

The overtrained 7B costs more to train but is **much cheaper to serve** than a 13B model with the same loss.

### LLaMA Strategy

| Model | Parameters | Training Tokens | Chinchilla Optimal | Ratio |
|---|---|---|---|---|
| LLaMA-1 7B | 7B | 1T | 140B | 7x overtrained |
| LLaMA-1 13B | 13B | 1T | 260B | 4x overtrained |
| LLaMA-1 65B | 65B | 1.4T | 1.3T | ~1x (Chinchilla-optimal) |
| LLaMA-2 7B | 7B | 2T | 140B | 14x overtrained |
| LLaMA-3 8B | 8B | 15T | 160B | 94x overtrained |

### The Economics

```
Training cost (one-time):    C_train = 6 * N * D_train
Inference cost (per token):  C_infer ≈ 2 * N

If you serve 10^12 tokens/year:
  7B model:  2 * 7B * 10^12 = 1.4 * 10^22 FLOPs/year
  70B model: 2 * 70B * 10^12 = 1.4 * 10^23 FLOPs/year (10x more)
```

Overtraining a 7B model costs more upfront but saves 10x on inference — and inference is the dominant cost in production.

### When to Overtrain vs Scale Up

| Factor | Overtrain Small | Scale Up |
|---|---|---|
| Inference volume | High → overtrain | Low → scale up |
| Latency requirement | Strict → small model | Relaxed → large OK |
| Training budget | Large (can afford overtraining) | Limited |
| Data availability | Need lots of data | Less data needed |

---

## Compute-Optimal vs Inference-Optimal

### Compute-Optimal (Chinchilla)

Minimize training loss for a fixed **training compute** budget:

```
minimize L(N, D) subject to 6*N*D = C_train
→ N_optimal ≈ C_train^0.5, D_optimal ≈ C_train^0.5
```

### Inference-Optimal (LLaMA Strategy)

Minimize **total cost** (training + inference over model lifetime):

```
minimize C_total = C_train + C_infer * V
         = 6*N*D + 2*N*V

Where V = total inference tokens over model lifetime
```

When `V` is large (production serving), the optimal `N` is **smaller** than Chinchilla-optimal — you overtrain a smaller model because inference dominates total cost.

---

## Data Quality and Scaling

### Data Quality > Data Quantity (Beyond a Point)

```
1T tokens of web scrape < 500B tokens of curated, filtered data
```

### Data Pipeline Quality Stages

1. **Web scraping**: Common Crawl, raw HTML
2. **Filtering**: remove spam, low-quality, duplicates
3. **Deduplication**: exact + fuzzy dedup (MinHash)
4. **Quality scoring**: classifier-based quality filtering
5. **Mixing**: balance domains (web, books, code, academic)
6. **Curated data**: Wikipedia, textbooks, code repositories

### Data Mixing

```
Typical mix (LLaMA-2):
  Web:    ~67%
  Code:   ~5%
  Books:  ~5%
  Academic: ~5%
  Other:  ~18%
```

The mix matters — too much web data degrades reasoning; too much code degrades natural language.

### Data Wall

```
High-quality tokens available: ~10-20T (after dedup and filtering)
LLaMA-3 trained on: 15T tokens
Future models may exhaust high-quality data → synthetic data becomes critical
```

---

## L5 Interview Q&A

### Q: "Why was GPT-3 considered undertrained?"

GPT-3 has 175B parameters but was trained on only 300B tokens. Chinchilla's optimal is ~20 tokens/parameter → 175B * 20 = 3.5T tokens. GPT-3 used ~12x less data than optimal. This is why smaller, properly trained models (LLaMA 7B on 2T tokens) outperform GPT-3.

### Q: "How would you decide the model size and training budget for a new LLM?"

1. **Determine inference budget**: how many tokens/year will you serve? This drives model size.
2. **Choose model size**: pick the largest model that fits inference latency/memory constraints.
3. **Compute training budget**: `C_train = 6 * N * D`. Choose `D` based on Chinchilla (20*N) or overtraining factor.
4. **Data availability**: ensure you have enough high-quality data for the chosen `D`.
5. **Total cost**: `C_total = C_train + C_infer * V`. Optimize for total, not just training.

### Q: "What is the data wall and how do you address it?"

The data wall: high-quality public text data is finite (~10-20T tokens after dedup). As models train on more tokens, they'll exhaust this supply.

Solutions:
1. **Synthetic data**: generate training data with existing LLMs (e.g., GPT-4 outputs).
2. **Multilingual data**: tap into non-English sources.
3. **Code data**: code is high-quality and abundant.
4. **Transcription**: convert audio/video to text.
5. **Curriculum learning**: train on higher-quality data later in training.

### Q: "Are emergent abilities real or a measurement artifact?"

**Both perspectives have merit:**
- **Real**: some capabilities (chain-of-thought reasoning, in-context learning) appear suddenly at certain scales and can't be predicted from smaller models.
- **Artifact**: some "emergence" is an artifact of evaluation metrics. If you measure exact-match accuracy, performance jumps from 0% to 100% at a threshold. If you measure log-likelihood, the improvement is smooth.
- **Current consensus**: the underlying capability improves smoothly, but the **evaluation metric** creates a threshold effect. The capability is real, but the sudden appearance is partly a measurement artifact.

### Q: "How does MoE change the scaling law calculus?"

MoE separates **parameters** from **compute**:
- A 67B MoE (e.g., Mixtral) has 67B parameters but only uses ~13B per token (2 experts active).
- Training compute scales with **active parameters**, not total.
- Inference compute also scales with active parameters.
- **Result**: you can have more total parameters (more capacity) without proportional compute increase.

This means MoE can achieve better loss than a dense model at the same compute budget — the scaling law exponents are more favorable.

---

## Interview Sound Bites

- Scaling laws: loss follows power laws with model size (N), data (D), and compute (C).
- **Chinchilla**: parameters and tokens should scale equally — ~20 tokens per parameter.
- **Kaplan (GPT-3)**: scaled parameters faster than data — suboptimal, led to undertraining.
- **Overtraining**: modern small models (LLaMA-3 8B on 15T tokens) are intentionally overtrained for inference efficiency.
- **Inference-optimal ≠ compute-optimal**: when inference volume is high, overtrain a smaller model — total cost is lower.
- FLOPs estimate: `C ≈ 6 * N * D` for training, `C ≈ 2 * N` per inference token.
- **Data quality > quantity**: curated, deduplicated, well-mixed data beats raw web scrape.
- **Data wall**: high-quality public text is finite (~10-20T tokens) — synthetic data is the future.
- **Emergent abilities**: underlying capability improves smoothly; sudden appearance is partly a metric artifact.
- **MoE**: separates parameters from compute — better scaling law exponents than dense models.
