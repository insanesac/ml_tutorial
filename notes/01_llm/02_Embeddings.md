# Embeddings: From One-Hot to BERT

## The Big Picture

Every new embedding technique exists because the previous one had a limitation:

```
Integer Encoding
  │  False numerical relationships
  ▼
One-Hot Encoding
  │  Huge, sparse vectors, no semantic similarity
  ▼
Word2Vec
  │  Learns semantic similarity, only local context
  ▼
GloVe
  │  Uses global corpus statistics, still one embedding per word
  ▼
FastText
  │  Uses character n-grams, handles rare/OOV words, still static
  ▼
BERT / GPT
  │  Contextual embeddings, different representation depending on sentence
```

---

## What is an Embedding?

Neural networks cannot operate directly on discrete token IDs.

```
"The cat sat" → [17, 824, 531]
```

Token IDs have no mathematical meaning. `cat = 17` and `dog = 18` does not imply dogs are similar to cats.

Instead, we convert each token into a dense vector:

```
cat → [0.31, -0.82, 1.17, ...]
```

This vector is called an **embedding**.

> **Interview Definition:** An embedding is a dense, continuous vector representation of a discrete token that captures useful semantic and syntactic information and serves as the numerical input to neural networks.

---

## Why Do We Need Embeddings?

Transformers perform operations like `Q @ K.T` or `W @ x`. These require floating-point vectors. Integers cannot be multiplied meaningfully.

```
Token ID → Embedding → Transformer
```

---

## 1. Integer Encoding

```
cat = 1, dog = 2, car = 3
```

**Problem:** The IDs are arbitrary. The neural network assumes `3 > 2 > 1`, that `dog` is somehow between `cat` and `car`, and that arithmetic differences have meaning. None of these are true. These are identifiers, not representations.

---

## 2. One-Hot Encoding

```
cat = [1 0 0]
dog = [0 1 0]
car = [0 0 1]
```

**What problem did it solve?** Removed the false numerical ordering. Each word becomes independent.

**New Problems:**
- **Huge vectors** — vocabulary = 50,000, every vector has length 50,000
- **Sparse** — almost every value is zero
- **No semantics** — all words are equally distant. The model has no notion that `cat ≈ dog`

---

## 3. Word2Vec

### Core Question

How can the model discover semantic similarity automatically?

Researchers observed: **words appearing in similar contexts usually have similar meanings.** This is called the **Distributional Hypothesis**.

> "You shall know a word by the company it keeps."

### Training Objective

Instead of manually defining similarity, predict neighboring words.

**Skip-Gram:** Predict context words from the target word.

```
The cat drank milk
cat → The, drank, milk
```

**CBOW:** Predict the target word from context.

```
The ___ drank milk → cat
```

### Why does this work?

Words appearing in similar contexts receive similar gradient updates.

```
king  → ruled, kingdom, crown
queen → ruled, kingdom, crown
```

Since the contexts are similar, the embedding vectors gradually become similar.

**Amazing Insight:** Word2Vec is NOT trying to learn similarity. It is trying to solve a prediction task. Semantic similarity emerges naturally.

### Sliding Window

Word2Vec uses a sliding window (same concept as the Sliding Window algorithm in DSA):

```
Sentence: The cat drank milk yesterday

Window: [The cat drank]
  Training pairs: (cat → The), (cat → drank)

Slide: [cat drank milk]
  Training pairs: (drank → cat), (drank → milk)
```

### Why nn.Embedding is just a lookup table

```
Input: cat → One-hot vector
One-Hot × W = simply selects one row of W
Therefore: Embedding(token) = W[token]
```

This is exactly what `nn.Embedding` does.

### Embedding Matrix

- **Initially:** `W = Random numbers`
- **Training:** `Loss → Backpropagation → Gradient → Update W`
- **Eventually:** `W = Meaningful embeddings`

---

## 4. GloVe (Global Vectors)

Word2Vec learns from local prediction. GloVe asks: across the entire corpus, how often do words occur together?

| Word | Word | Count |
|---|---|---|
| king | crown | 120,000 |
| king | queen | 90,000 |
| king | bicycle | 2 |

These global co-occurrence statistics contain semantic information.

**Limitation:** Still learns one word → one embedding.

---

## 5. FastText

### Problem

`computer`, `computerized`, `computerization` — Word2Vec treats these as unrelated words.

### Key Insight

Words are made of smaller pieces. Instead of learning only word embeddings, learn embeddings for **character n-grams**.

```
computer → 3-grams: <com, com, omp, mpu, put, ute, ter, ter>
```

Each n-gram has an embedding. The final word embedding is formed by combining these subword embeddings.

### Advantages

- Handles **rare words**, **unknown words (OOV)**, and **morphological variations**
- Even unseen words can be represented

### Why memory doesn't explode

The same n-grams are reused across many words:

```
computer, computers, computerized, computing
→ all reuse: comp, omp, mpu, put, ute, ...
```

Embeddings are shared.

### Limitation

FastText still gives `bank → One embedding`. So "river bank" and "finance bank" still share exactly the same vector.

---

## 6. Contextual Embeddings (BERT/GPT)

### Static embeddings

```
bank → One vector (always the same)
```

### Contextual embeddings

```
"I deposited money in the bank"  → bank_finance
"The fisherman sat on the bank"  → bank_river
```

The initial embedding is identical. **Self-attention transforms it** into different contextual representations.

### Complete Pipeline

```
Token ID → Embedding Layer → Static Embedding → Self Attention → Feed Forward → Self Attention → ... → Contextual Representation
```

The embedding is merely the starting point. The final vectors produced after many Transformer layers are **contextual representations**.

---

## Vector Arithmetic

One famous discovery:

```
king - man + woman ≈ queen
```

**Interpretation:** King → remove male → royal person → add female → queen.

No dimension explicitly represents "royalty" or "gender." These directions emerge naturally during training.

---

## What is nn.Embedding?

Suppose:
- Vocabulary Size = 50,000
- Embedding Dimension = 768

The embedding layer stores a matrix of shape `50000 × 768`. Each row corresponds to one token.

| Token ID | Token | Embedding |
|---|---|---|
| 0 | `<PAD>` | [...] |
| 17 | cat | [...] |
| 824 | dog | [...] |
| 531 | sat | [...] |

When we execute `embedding(token_ids)`, PyTorch simply performs:

```python
embedding.weight[token_ids]  # indexing rows
```

There is:
- No matrix multiplication
- No neural network computation
- No attention

It is simply **indexing rows**.

> **Interview Answer:** `nn.Embedding` is a trainable lookup table. Given token IDs, it retrieves the corresponding embedding vectors by indexing into the embedding matrix.

---

## How Are Embeddings Learned?

The embedding matrix is not fixed. It is a **trainable parameter**.

```
dog → Embedding → Transformer Layers → Cross Entropy Loss → Backpropagation → Update Embedding(dog)
```

During training, gradients flow all the way back to `embedding.weight`, just like every other weight matrix. The embedding vectors gradually improve throughout training.

> **Interview Answer:** Embeddings are learned through backpropagation. Since the embedding matrix is a trainable parameter, gradients update each token's vector based on the prediction error.

---

## Why Do Similar Words End Up Close Together?

Suppose the corpus contains:

```
The dog ran.     The cat ran.
The dog slept.   The cat slept.
The dog barked.  The cat meowed.
```

Both `dog` and `cat` appear in similar contexts. During training, they receive very similar gradient updates. Their vectors move through embedding space in similar directions.

Eventually: `CosineSimilarity(cat, dog) ≈ High`

Nobody explicitly tells the model "put cat near dog." It emerges naturally from training.

> **Interview Answer:** Words occurring in similar contexts receive similar gradient updates during training, causing their embeddings to become close in the vector space.

---

## Common Interview Q&A

### Q: Why does BERT produce different representations for "bank" even though nn.Embedding always returns the same vector?

The embedding layer always returns the same static embedding for a token. BERT then applies multiple layers of bidirectional self-attention, allowing each token to interact with the surrounding context. These interactions transform the original embedding into a contextual representation whose meaning depends on the sentence.

### Q: What is the difference between Word2Vec and GloVe?

Word2Vec learns from **local context** using a sliding window prediction task (skip-gram or CBOW). GloVe uses **global co-occurrence statistics** across the entire corpus. Both produce static embeddings — one vector per word regardless of context.

### Q: How does FastText handle out-of-vocabulary words?

FastText decomposes words into character n-grams. Each n-gram has its own embedding. For an unseen word, the model combines the embeddings of its constituent n-grams to produce a representation. This is why FastText can handle rare and OOV words that Word2Vec cannot.

---

## Key Takeaways

- Integer encoding creates false numerical relationships.
- One-hot encoding fixes ordering but produces huge, sparse, semantically empty vectors.
- Word2Vec learns semantic similarity through local context prediction (skip-gram/CBOW). Similarity emerges naturally, not by design.
- GloVe uses global co-occurrence statistics instead of local prediction.
- FastText uses character n-grams to handle OOV and rare words, but still produces static embeddings.
- `nn.Embedding` is a trainable lookup table (row indexing, not computation).
- The embedding matrix is updated through backpropagation.
- Similar words become close because they appear in similar contexts → similar gradient updates.
- The embedding layer always returns the **same** vector for a token (static).
- Contextual representations are created **after** multiple layers of self-attention (BERT/GPT).
- BERT and GPT do not learn different embeddings for the same word; they learn different **contextual representations** from the same initial embedding.

---

## One-Sentence Evolution

```
IDs → Independent Words → Semantic Words → Global Semantics → Subword Semantics → Contextual Semantics
```
