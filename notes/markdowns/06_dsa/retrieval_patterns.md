# ML/LLM Coding Patterns — Retrieval, Ranking & Context

Most LLM-focused coding interviews test data processing, retrieval, ranking, evaluation, context management, and vector operations — not transformer implementation.

---

## Cosine Similarity

**Keywords:** `embeddings`, `similarity`, `semantic search`, `retrieval`, `nearest neighbor`

```text
dot(A, B)
---------
|A| * |B|
```

```python
import math

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
```

Complexity: O(D) where D = embedding dimension

---

## Dot Product

Fast retrieval scoring (when vectors are already normalized).

```python
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))
```

Complexity: O(D)

---

## Euclidean Distance

```python
import math

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
```

Complexity: O(D)

---

## Top-K Retrieval

**Pattern:** Similarity + Heap

```python
import heapq

def top_k_retrieval(query_embedding, docs, k):
    heap = []
    for doc_id, embedding in docs.items():
        score = cosine_similarity(query_embedding, embedding)
        heapq.heappush(heap, (score, doc_id))
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted(heap, reverse=True)
```

Complexity: O(ND + N log K) — similarity computation + heap maintenance

---

## Chunking

**Keywords:** `context window`, `token limit`, `split documents`

### Fixed Chunk Size

```python
def chunk_text(words, chunk_size):
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(words[i:i + chunk_size])
    return chunks
```

### Sliding Window Chunking (with overlap)

```python
start = 0
while start < len(tokens):
    end = start + chunk_size
    chunk = tokens[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap
```

**Edge case:** `overlap >= chunk_size` → infinite loop

Complexity: O(N)

---

## Context Window Management

**Problem:** Keep newest messages under token budget.

**Pattern:** Greedy reverse traversal

```python
total_tokens = 0
selected = []

for msg in reversed(messages):
    if total_tokens + msg.tokens > token_limit:
        break
    selected.append(msg)
    total_tokens += msg.tokens

selected.reverse()
```

Complexity: O(N)

---

## Deduplication

### Exact Duplicate

```python
seen = set()
result = []
for chunk in chunks:
    if chunk not in seen:
        seen.add(chunk)
        result.append(chunk)
```

### Near-Duplicate Detection

```python
if cosine_similarity(emb1, emb2) > 0.95:
    # treat as duplicate
```

Complexity: O(N) for exact, O(N²D) for near-duplicate pairwise

---

## Retrieval Metrics

### Precision

```text
TP / (TP + FP)
```

### Recall

```text
TP / (TP + FN)
```

### F1

```text
2PR / (P + R)
```

### Recall@K

```text
Relevant Retrieved in top K / Total Relevant
```

```python
def recall_at_k(ranked, relevant, k):
    top_k = set(ranked[:k])
    return len(top_k & relevant) / len(relevant)
```

### MRR (Mean Reciprocal Rank)

```text
1 / rank of first relevant document
```

```python
def reciprocal_rank(ranked_docs, relevant_docs):
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in relevant_docs:
            return 1 / rank
    return 0
```

### NDCG

Measures ranking quality. Rewards relevant docs appearing early. Understand concept — usually not derived from memory in interviews.

---

## Reranking

**Pattern:** Score → Sort → Return Top K

```python
scores = []
for doc in docs:
    score = cosine_similarity(query_embedding, doc.embedding)
    scores.append((score, doc))
scores.sort(reverse=True)
```

---

## Pattern Flow Summary

| Problem | Pattern |
|---|---|
| Semantic search | Embedding → Similarity → Top K (Heap) |
| RAG retrieval | Query → Embed → Similarity → Top K → Rerank → Context |
| Context management | Reverse traversal + token budget |
| Chunking | Sliding window with overlap |
| Deduplication | Set (exact) / Similarity threshold (near-dup) |
| Ranking evaluation | Recall@K, MRR, NDCG |

---

## Last-Minute Checklist

Can I implement:
- [x] Cosine Similarity
- [x] Dot Product
- [x] Top K Retrieval (with heap)
- [x] Chunking (fixed + sliding window)
- [x] Context Truncation
- [x] Precision / Recall / F1
- [x] Recall@K
- [x] MRR

If yes, you can handle a large fraction of LLM-focused ML coding interviews.
