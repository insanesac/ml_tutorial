# Probability, Information Theory & Statistics

## The Big Picture

These three fields answer different questions:

- **Probability:** What is our belief about an outcome before observing it?
- **Information Theory:** How much do we learn when an event occurs?
- **Statistics:** How should beliefs change after observing new evidence?

---

## 1. Probability

### Definition

Probability represents our **belief** about an outcome before observing it. It is not the outcome itself.

```
Fair Die: P(1) = 1/6, P(2) = 1/6, ..., P(6) = 1/6
```

**Properties:**
- `0 ≤ P(x) ≤ 1`
- Sum of probabilities = 1

A collection of probabilities over all outcomes is called a **Probability Distribution**.

---

## 2. Information (Surprise)

### Question

How much information do we gain when an event occurs?

### Observation

- Common events carry **little** information
- Rare events carry **lots** of information

### Requirements for an information measure

1. Probability decreases → Information increases
2. Certain event (P=1) → Information = 0
3. Independent events should have **additive** information

Since probabilities multiply while information should add, logarithm naturally appears.

### Formula

```
I(x) = -log(P(x))
```

**Interpretation:**
- High probability → Low surprise
- Low probability → High surprise

---

## 3. Entropy

### Question

How surprised are we **on average**?

Entropy is simply the **expected surprise**:

```
H(P) = -Σ P(x) log(P(x))
```

**Interpretation:** Average uncertainty of a probability distribution.

### Examples

| Distribution | Entropy |
|---|---|
| Fair coin (0.5, 0.5) | High |
| Biased coin (0.999, 0.001) | Low |

Higher entropy means greater uncertainty.

---

## 4. Cross Entropy

Entropy measures surprise using the **true** distribution. Cross Entropy measures surprise using the **model's predicted** distribution.

```
H(P,Q) = -Σ P(x) log(Q(x))
```

Where:
- `P` = True distribution
- `Q` = Model prediction

**Interpretation:** Average surprise experienced by the model.

In classification with one-hot labels:

```
Loss = -log(predicted probability of the correct class)
```

**Why `y * log(pred)`?** The true label acts as a selector. Only the correct class contributes to the loss.

---

## 5. KL Divergence

### Question

How much **additional** surprise do we experience because our beliefs are incorrect?

```
KL(P||Q) = CrossEntropy(P,Q) - Entropy(P)
```

**Interpretation:**

```
Cross Entropy = Entropy + KL Divergence
```

**Properties:**
- `KL ≥ 0`
- `KL = 0` only when `P = Q`

**Intuition:** KL Divergence is the "cost of having incorrect beliefs."

---

## 5b. Mutual Information

### Question

Does knowing one variable tell me something about the other?

This is about **variables**, not events.

### Example

```
Variable 1: Height
Variable 2: Weight

If I tell you someone is 220 cm tall...
Does that help you guess their weight?
Yes → Positive mutual information.
```

### Another Example

```
Variable A: Outside Temperature
Variable B: Ice Cream Sales

Hot day → You'd expect higher ice cream sales.
Temperature ↔ Ice Cream Sales → High mutual information.
But they're certainly not mutually exclusive — they happen together!
```

### Mutually Exclusive vs Independent vs Mutual Information

These are three completely different ideas sharing the word "mutual":

| Concept | Question |
|---|---|
| Mutually Exclusive | Can these two events happen together? |
| Independent | Does one event affect the probability of the other? |
| Mutual Information | Does knowing one variable reduce uncertainty about the other? |

### Coin Toss Example

```
Event A: Heads
Event B: Tails
```

- **Mutually exclusive?** Yes — they can't both happen on the same toss.
- **Independent?** No — if you know it's Heads, it definitely isn't Tails.
- **Mutual information?** Knowing one completely determines the other, so they share information.

Don't let the word "mutual" fool you — it's just an English word reused in different contexts.

### Relationship to KL Divergence

Mutual information `I(X; Y)` measures how much knowing Y reduces uncertainty about X. It can be expressed as the KL divergence between the joint distribution and the product of marginals:

```
I(X; Y) = KL(P(X,Y) || P(X)·P(Y))
```

If X and Y are independent, `P(X,Y) = P(X)·P(Y)`, so `I(X; Y) = 0` — knowing Y tells you nothing about X.

---

## 6. Perplexity

Cross entropy is difficult to interpret. Researchers convert it into an equivalent number of equally likely choices.

```
Perplexity = exp(Cross Entropy)
```

**Interpretation:** Average number of equally likely choices the model is confused between.

| Perplexity | Meaning |
|---|---|
| 2 | Model behaves as if choosing between ~2 next tokens |
| 100 | Model is highly uncertain |

**Relationship:**

```
Lower Cross Entropy → Lower Perplexity → Better language model
```

---

## 7. Conditional Probability

### Definition

Probability of A after learning that B is true.

```
P(A|B) = P(A ∩ B) / P(B)
```

**Interpretation:** Conditional probability **shrinks the universe**.

### Example

```
52 cards → Given card is a Spade → Universe becomes 13 cards

P(Ace of Spades | Spade) = 1 / 13
```

The numerator stays the same. The denominator changes because impossible worlds are discarded.

---

## 8. Bayes' Theorem

Derived directly from conditional probability:

```
         P(B|A) P(A)
P(A|B) = ─────────────
              P(B)
```

**Interpretation:** Bayes **reverses** probability. It converts:

- "What is the probability of observing the evidence given the cause?"
- into "What is the probability of the cause given the observed evidence?"

### Examples

```
Disease → Symptoms    (reality runs cause → effect)
Fire → Smoke
Rain → Umbrella

Inference runs: Effect ↑ → Cause?
```

---

## 9. Expectation

### Question

What happens **on average**?

```
E[X] = Σ P(x) · X
```

**Interpretation:** Weighted average of all possible outcomes.

**Connection:** Entropy itself is simply the expectation of surprise:

```
Entropy = E[-log(P)]
```

---

## 10. Variance

Expectation tells us the average. Variance tells us **how spread out** values are.

Naively averaging deviations fails because positives and negatives cancel. Solution: **square** deviations.

```
Var(X) = E[(X - μ)²]
```

**Interpretation:** Average squared distance from the mean. Higher variance means greater spread.

---

## 11. Standard Deviation

Variance has squared units (e.g., height variance = cm²). Standard deviation restores the original units.

```
StdDev = √Variance
```

**Interpretation:** Typical deviation from the mean.

---

## 12. Covariance

### Question

Do two variables move together?

```
Cov(X,Y) = E[(X - μx)(Y - μy)]
```

| Covariance | Meaning |
|---|---|
| Positive | Both variables increase together |
| Negative | One increases while the other decreases |
| Near zero | No linear relationship |

**Relationship:** Variance is simply covariance with itself:

```
Var(X) = Cov(X, X)
```

---

## 13. Correlation

### Problem

Covariance depends on measurement units.

### Solution

Normalize covariance:

```
Correlation = Cov(X,Y) / (σx · σy)
```

| Value | Meaning |
|---|---|
| +1 | Perfect positive relationship |
| 0 | No linear relationship |
| -1 | Perfect negative relationship |

---

## 14. Gaussian Distribution

### Question

Why do so many natural phenomena follow a bell curve?

### Answer

Because many independent random factors combine together. This is explained by the **Central Limit Theorem**.

### Examples

Height, IQ, sensor noise, measurement errors, weight initialization.

### Parameters

The Gaussian is fully described by only two parameters:

| Parameter | Role |
|---|---|
| Mean (μ) | Determines the center |
| Standard Deviation (σ) | Determines the width |

**Notation:** `N(μ, σ²)`

### 68-95-99.7 Rule

| Range | % of values |
|---|---|
| ±1σ | ~68% |
| ±2σ | ~95% |
| ±3σ | ~99.7% |

### Applications

Diffusion models (Gaussian noise), weight initialization, measurement errors, Kalman filters, Gaussian processes, data normalization.

---

## Key Interview Takeaways

| Concept | One-line |
|---|---|
| Probability | Belief before observing an outcome |
| Information | Surprise (`-log(P)`) |
| Entropy | Average surprise |
| Cross Entropy | Average surprise using model's predicted probabilities |
| KL Divergence | Extra surprise caused by incorrect beliefs |
| Mutual Information | Does knowing one variable reduce uncertainty about the other? |
| Perplexity | Average number of equally likely choices the model considers |
| Conditional Probability | Shrink the universe after observing evidence |
| Bayes' Theorem | Reason backwards from observed effects to possible causes |
| Expectation | Weighted average |
| Variance | Average squared deviation from the mean |
| Standard Deviation | Typical deviation in original units |
| Covariance | Do two variables move together? |
| Correlation | Normalized covariance (unitless, -1 to +1) |
| Gaussian | Distribution produced by many independent random effects |
