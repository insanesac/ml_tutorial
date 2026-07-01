# Sequence Models: From Feed Forward Networks to Attention

## The Big Picture

Every new architecture was invented to solve a limitation of the previous one:

```
Feed Forward Network
  ↓  No memory, fixed-size inputs
RNN
  ↓  Vanishing gradients, sequential computation
LSTM
  ↓  Still sequential, fixed memory capacity
GRU
  ↓  Simplified LSTM
Seq2Seq
  ↓  Entire sentence compressed into one context vector
Attention
  ↓  Removes context vector bottleneck
Transformer (covered separately)
```

---

## 1. Feed Forward Networks

### Strengths

Excellent for image classification, regression, tabular data — any fixed-size input.

```
Image → Cat / Dog
House Features → House Price
```

### Fundamental Limitation

Feed Forward Networks have **no memory**. Each prediction depends only on the current input.

- Fixed input size
- No notion of sequence
- No notion of time

```
"Dog bites man."  vs  "Man bites dog."
```

Both contain identical words but completely different meanings. A feed-forward network has no natural mechanism to understand ordering.

### Problem Statement

How do we build a neural network that **remembers** previous words?

---

## 2. Recurrent Neural Network (RNN)

### Idea

Instead of processing only the current word, also feed the previous hidden state back into the network.

```
Instead of:  h = f(x)
We compute:  hₜ = f(xₜ, hₜ₋₁)

where:
  xₜ    = current word
  hₜ₋₁  = previous memory
```

The hidden state becomes a **compressed summary** of everything seen so far.

### Advantages

- Introduces memory
- Can process variable-length sequences
- Suitable for language, speech, time series

### Problems

**1. Information Bottleneck**

Only one hidden state stores the entire history. Every new word updates the same memory. Eventually older information gets overwritten.

```
Trying to summarize a tweet and a novel using the same-sized memory.
```

**2. Vanishing Gradient**

During training, the gradient must travel backwards through many time steps. For long sequences, learning signals gradually disappear.

```
"The movie was not ...(100 words later)... good."
The model often forgets the word "not".
```

**3. Sequential Computation**

Each hidden state depends on the previous one:

```
h1 → h2 → h3 → h4
```

Computation cannot be parallelized. GPUs remain largely underutilized.

---

## 3. Long Short-Term Memory (LSTM)

### Motivation

RNN has memory, but it has no control over **what to remember** and **what to forget**. Everything is mixed together.

### Main Idea

Introduce **controlled memory**. Instead of blindly updating memory, learn what to forget, what to remember, and what to expose.

### Components

| Component | Role |
|---|---|
| **Cell State** | Long-term memory. Carries important information across many time steps. |
| **Hidden State** | Working memory. Used for the current prediction. |

### Gates

| Gate | Question | Function |
|---|---|---|
| **Forget Gate** | "What should I forget?" | Removes unnecessary information |
| **Input Gate** | "What should I remember?" | Stores important new information |
| **Output Gate** | "What should I reveal?" | Controls which information is exposed to the next layer |

### Advantages

- Greatly reduces vanishing gradients
- Learns much longer dependencies
- Controlled memory updates

### Remaining Problems

- Still sequential
- Still uses a fixed-size memory representation
- Still difficult to parallelize

---

## 4. Gated Recurrent Unit (GRU)

### Observation

LSTM works well, but is fairly complicated.

### Idea

Simplify LSTM:

- Removes the separate cell state
- Uses fewer gates
- Fewer parameters
- Faster training

### Interview Summary

| | LSTM | GRU |
|---|---|---|
| Expressiveness | More expressive | Simpler |
| Long dependencies | Better | Often similar |
| Speed | Slower | Faster |
| Parameters | More | Fewer |

---

## 5. Sequence-to-Sequence (Seq2Seq)

### Goal

Translate one sequence into another.

```
Input Sentence
  ↓
Encoder RNN
  ↓
Context Vector
  ↓
Decoder RNN
  ↓
Output Sentence
```

The encoder converts the input sentence into a single **context vector**. The decoder generates the output sentence from that vector.

### Problem

Entire sentence → One vector. Large information bottleneck. Long sentences lose important information.

```
Trying to compress an entire paragraph into a single vector.
```

---

## 6. Attention

### Key Insight

The decoder no longer depends only on one context vector. Instead, it has access to **every encoder hidden state**.

While generating each output token, the decoder decides which encoder words deserve attention.

### Example

```
Input: "The movie was not good."

Generating "good":
  The model attends strongly to "not"
  instead of treating every previous word equally.
```

### Advantages

- Removes the single context bottleneck
- Long sentences become much easier to translate
- The decoder can dynamically retrieve information from any part of the input

### Why Attention Changed Everything

```
Instead of:  "I must remember everything."
Attention:   "I'll simply look at what I need."
```

This idea eventually led to Transformers.

---

## Evolution Summary

| Architecture | Innovation | Problem Solved | Remaining Problem |
|---|---|---|---|
| Feed Forward | Fixed-size mapping | Basic classification | No memory |
| RNN | Recurrent hidden state | Memory for sequences | Vanishing gradients, sequential |
| LSTM | Gates (forget/input/output) | Long-term dependencies | Still sequential |
| GRU | Simplified LSTM | Fewer parameters | Same limitations |
| Seq2Seq | Encoder-decoder | Sequence translation | Context vector bottleneck |
| Attention | Dynamic retrieval | Removes bottleneck | Still uses recurrence |
| Transformer | Self-attention | Removes recurrence | (covered separately) |

---

## Interview Takeaways

- **Feed Forward:** No memory. Fixed-size inputs only.
- **RNN:** Introduces memory using recurrent hidden states. Main problems: vanishing gradients, information bottleneck, sequential execution.
- **LSTM:** Controlled memory through forget, input, and output gates. Greatly improves long-term dependency learning.
- **GRU:** Simpler, faster version of LSTM with fewer parameters.
- **Seq2Seq:** Encoder-decoder architecture for sequence generation. Limited by a single context vector.
- **Attention:** Allows the decoder to dynamically retrieve information from any encoder hidden state. Eliminates the context bottleneck and lays the foundation for Transformers.
