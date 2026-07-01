# Advanced RAG Techniques

## Production RAG Pipeline

```
User Query
  │
  ▼
Metadata Filtering
  │
  ▼
Query Expansion / Multi-Query / HyDE
  │
  ▼
Vector Retrieval
  │
  ▼
Parent-Child Retrieval
  │
  ▼
Context Compression
  │
  ▼
Final LLM
```

---

## 1. Metadata Filtering

### Motivation

Searching every document in a vector database is often unnecessary. Documents usually contain structured metadata.

```json
{
    "author": "Sachin",
    "department": "AI",
    "year": 2026,
    "document_type": "Design Doc",
    "security": "Internal"
}
```

### Solution

Apply structured constraints **before** semantic retrieval:

```
Query: "Show me AI design documents from 2026"
Filter: department = AI AND year = 2026
→ Search matching documents only
```

### Common Metadata Fields

Author, Date, Department, Language, Project, Security Level, Document Type

### Benefits

- Higher precision
- Lower latency (smaller search space)
- Better access control

---

## 2. Parent-Child Retrieval

### Motivation

| Chunk Size | Retrieval | Context |
|---|---|---|
| Small | Good | Poor (lacks context) |
| Large | Poor | Good (rich context) |

### Solution

```
Store Large Parent Document
  ↓ Split Into
Small Child Chunks
```

- Embeddings created **only for child chunks**
- Each child stores a reference to its parent
- **Retrieve** child chunk → **Return** entire parent (or neighboring chunks) to LLM

### Benefits

- Accurate retrieval (small chunks match well)
- Rich generation context (parent provides full picture)
- Best balance between precision and context

---

## 3. Context Compression

### Motivation

Vector retrieval often returns much more text than necessary:

```
Retrieved:           5000 tokens
Useful Information:   400 tokens
```

The remaining tokens increase cost, latency, and hallucination risk.

### Solution

Compress retrieved context before sending to LLM:

```
Retrieve Top-K → Context Compression → Relevant Sentences → LLM
```

Compression methods: another LLM, extractive summarization, rerankers, sentence filtering.

### Benefits

- Lower token usage
- Faster inference
- Lower cost
- Better signal-to-noise ratio

---

## 4. Query Expansion

### Motivation

Users often ask incomplete or ambiguous questions. Relevant documents may use different terminology:

```
Query: "car engine"
Relevant docs: "automobile engine", "vehicle engine", "internal combustion engine"
```

### Solution

Automatically enrich the query:

```
Original:  "car engine"
Expanded:  "car engine", "automobile engine", "vehicle engine", "internal combustion engine"
```

### Benefits

- Improved recall
- Better semantic matching
- More robust retrieval

---

## 5. HyDE (Hypothetical Document Embeddings)

### Motivation

Short queries produce weak embeddings:

```
"What is ALiBi?" → Very little semantic information
```

### Solution

Ask an LLM to generate a hypothetical answer, then embed **that** instead of the query:

```
User Query
  ↓
Generate Hypothetical Answer (LLM)
  ↓
Embedding
  ↓
Vector Search
```

The hypothetical answer is never shown to the user — it exists only to improve retrieval.

### Why It Works

Embeddings capture semantic meaning much better from rich paragraphs than from short questions.

---

## 6. Multi-Query Retrieval

### Motivation

The same intent can be expressed many ways:

```
"Gradient Checkpointing"
= "Memory optimization during training"
= "Reducing GPU memory"
= "Activation recomputation"
```

One embedding cannot capture every semantic variation.

### Solution

Generate multiple versions of the query:

```
Original Query
  ↓ Generate Multiple Queries (LLM)
  ↓ Vector Search for Each
  ↓ Merge Results
  ↓ Remove Duplicates
  ↓ LLM
```

### Benefits

- Higher recall
- Better semantic coverage
- Finds documents missed by a single query

### Drawback

Instead of 1 search → 5 searches. Higher latency and cost.

---

## Which Stage Does Each Technique Improve?

| Technique | Stage Improved |
|---|---|
| Metadata Filtering | Candidate Selection |
| Query Expansion | Query Understanding |
| HyDE | Query Representation |
| Multi-Query Retrieval | Query Representation |
| Parent-Child Retrieval | Retrieved Context |
| Context Compression | Final Context Passed to LLM |

## Recall vs Precision

| Goal | Techniques |
|---|---|
| **Improve Recall** (find more relevant docs) | Query Expansion, HyDE, Multi-Query Retrieval |
| **Improve Precision** (return better docs) | Metadata Filtering, Parent-Child Retrieval, Context Compression |

---

## Interview Q&A

**The LLM hallucinates — what's wrong and how to fix it?**

| Symptom | Cause | Fix |
|---|---|---|
| LLM hallucinates | Retrieval missed relevant docs | Improve recall: Query Expansion, HyDE, Multi-Query |
| LLM hallucinates | Wrong documents retrieved | Improve precision: Metadata Filtering, reranking |
| LLM hallucinates | Retrieved chunks lack context | Parent-Child Retrieval |
| LLM hallucinates | Too much irrelevant context | Context Compression |

## Complete Comparison

| Technique | Main Goal | Main Benefit | Drawback |
|---|---|---|---|
| Metadata Filtering | Restrict search space | Higher precision | Requires structured metadata |
| Parent-Child Retrieval | Better context | Accurate retrieval + rich context | More storage and bookkeeping |
| Context Compression | Reduce unnecessary tokens | Lower cost, less hallucination | Additional compression step |
| Query Expansion | Expand search terms | Better recall | Slightly higher retrieval cost |
| HyDE | Richer semantic query | Better semantic retrieval | Extra LLM call |
| Multi-Query Retrieval | Explore multiple semantic views | Highest recall | Multiple retrieval operations |
