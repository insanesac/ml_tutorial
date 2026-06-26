# Google ML Coding Interview Cheat Sheet (LLM / Retrieval / ML Focused)

## Goal

Most LLM-focused coding interviews are not testing:

* Transformer implementation
* Backpropagation derivations
* CUDA kernels

Instead they test:

```text
Data Processing
Retrieval
Ranking
Evaluation
Context Management
Vector Operations
```

---

# 1. Cosine Similarity

## When

Keywords:

* Embeddings
* Similarity
* Semantic Search
* Retrieval
* Nearest Neighbor

---

## Formula

```text
dot(A,B)
---------
|A| * |B|
```

---

## Implementation

```python
import math

def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))

    return dot / (norm_a * norm_b)
```

---

## Complexity

```text
O(D)
```

D = embedding dimension

---

# 2. Dot Product

## When

Fast retrieval scoring.

---

## Implementation

```python
def dot_product(a, b):
    return sum(x*y for x,y in zip(a,b))
```

---

## Complexity

```text
O(D)
```

---

# 3. Euclidean Distance

## Formula

```text
sqrt(sum((a-b)^2))
```

---

## Implementation

```python
import math

def euclidean_distance(a, b):

    return math.sqrt(
        sum(
            (x-y)**2
            for x,y in zip(a,b)
        )
    )
```

---

# 4. Top-K Retrieval

## Problem

Given:

```python
query_embedding
document_embeddings
k
```

Return:

```text
Top K most similar documents
```

---

## Brute Force

Compute similarity against all documents.

Sort.

Return first K.

---

## Implementation

```python
scores = []

for doc_id, embedding in docs.items():

    score = cosine_similarity(
        query_embedding,
        embedding
    )

    scores.append((score, doc_id))

scores.sort(reverse=True)

return scores[:k]
```

---

## Complexity

```text
O(ND)
```

Similarity computation

plus

```text
O(N log N)
```

sorting

---

## Optimization

Heap:

```text
O(N log K)
```

---

# 5. Top-K Retrieval Using Heap

## Pattern

```text
Similarity
+
Top K
```

↓

```text
Heap
```

---

## Template

```python
import heapq

heap = []

for doc_id, score in scores:

    heapq.heappush(
        heap,
        (score, doc_id)
    )

    if len(heap) > k:
        heapq.heappop(heap)
```

---

# 6. Chunking

## When

Keywords:

* Context Window
* Token Limit
* Split Documents

---

## Fixed Chunk Size

```python
def chunk_text(words, chunk_size):

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):
        chunks.append(
            words[i:i+chunk_size]
        )

    return chunks
```

---

## Complexity

```text
O(N)
```

---

# 7. Sliding Window Chunking

## Why

Avoid cutting sentences abruptly.

---

Example

```text
Chunk 1:
1-500

Chunk 2:
450-950
```

Overlap:

```text
50 tokens
```

---

## Template

```python
start = 0

while start < len(tokens):

    end = start + chunk_size

    chunk = tokens[start:end]

    chunks.append(chunk)

    start += (
        chunk_size
        - overlap
    )
```

---

# 8. Context Window Management

## Problem

Given:

```python
messages
token_limit
```

Keep newest messages while respecting limit.

---

## Pattern

```text
Sliding Window
```

---

## Implementation

```python
total_tokens = 0

selected = []

for msg in reversed(messages):

    if (
        total_tokens
        + msg.tokens
        > token_limit
    ):
        break

    selected.append(msg)

    total_tokens += msg.tokens

selected.reverse()
```

---

# 9. Deduplication

## Problem

Remove duplicate chunks.

---

## Exact Duplicate

```python
seen = set()

result = []

for chunk in chunks:

    if chunk not in seen:

        seen.add(chunk)

        result.append(chunk)
```

---

## Complexity

```text
O(N)
```

---

# 10. Near-Duplicate Detection

## Pattern

```text
Similarity
+
Threshold
```

---

## Example

```python
if cosine_similarity(
    emb1,
    emb2
) > 0.95:
```

Treat as duplicate.

---

# 11. Precision

## Formula

```text
TP
---------
TP + FP
```

---

## Implementation

```python
def precision(tp, fp):

    return tp / (tp + fp)
```

---

# 12. Recall

## Formula

```text
TP
---------
TP + FN
```

---

## Implementation

```python
def recall(tp, fn):

    return tp / (tp + fn)
```

---

# 13. F1 Score

## Formula

```text
2PR
-----------
P + R
```

---

## Implementation

```python
def f1(p, r):

    return (
        2 * p * r
    ) / (p + r)
```

---

# 14. Recall@K

## Meaning

Did we retrieve the relevant document in top K?

---

## Formula

```text
Relevant Retrieved
-------------------
Total Relevant
```

---

## Example

Relevant:

```python
{1,2,3}
```

Retrieved:

```python
[1,4,5]
```

Recall@3:

```text
1 / 3
```

---

# 15. MRR

Mean Reciprocal Rank

---

## Formula

```text
1
---------
Rank
```

---

Example

Relevant document at:

```text
Rank 3
```

MRR:

```text
1/3
```

---

## Implementation

```python
def reciprocal_rank(
    ranked_docs,
    relevant_docs
):

    for rank, doc in enumerate(
        ranked_docs,
        start=1
    ):

        if doc in relevant_docs:
            return 1/rank

    return 0
```

---

# 16. NDCG

## Purpose

Measures ranking quality.

Rewards:

```text
Relevant docs
appearing early
```

---

## Keywords

* Ranking
* Search Quality
* Retrieval Evaluation

---

Interview expectation:

Understand concept.

Usually not derive from memory.

---

# 17. Simple Reranking

## Problem

Retrieved documents.

Need better ordering.

---

## Pattern

```text
Score
Sort
Return Top K
```

---

## Implementation

```python
scores = []

for doc in docs:

    score = cosine_similarity(
        query_embedding,
        doc.embedding
    )

    scores.append(
        (score, doc)
    )

scores.sort(
    reverse=True
)
```

---

# 18. Embedding Search Pattern

## Recognition

Keywords:

* Similarity Search
* Semantic Search
* Vector Search
* Nearest Neighbor

---

Pattern:

```text
Embedding
↓
Similarity Function
↓
Top K
↓
Heap / Sort
```

---

# 19. RAG Retrieval Pattern

## Recognition

Keywords:

* Retrieval
* Search
* Knowledge Base
* Documents

---

Pattern:

```text
Query
↓
Embedding
↓
Similarity
↓
Top K
↓
Rerank
↓
Context
```

---

# 20. Most Important ML Coding Patterns

| Problem Type        | Pattern            |
| ------------------- | ------------------ |
| Frequency Count     | HashMap            |
| Deduplication       | Set                |
| Top K Results       | Heap               |
| Retrieval           | Similarity + Heap  |
| Semantic Search     | Embedding + Top K  |
| Context Management  | Sliding Window     |
| Chunking            | Sliding Window     |
| Ranking             | Sort / Heap        |
| Recall@K            | Set Operations     |
| MRR                 | Ranking Scan       |
| RAG Retrieval       | Similarity + Top K |
| Conversation Memory | Sliding Window     |

---

# Last-Minute Interview Checklist

Can I implement:

- [x] Cosine Similarity

- [x] Dot Product

- [x] Top K Retrieval

- [x] Heap

- [x] Chunking

- [x] Sliding Window

- [x] Context Truncation

- [x] Precision

- [x] Recall

- [x] F1

- [x] Recall@K

- [x] MRR

If yes, I can handle a large fraction of LLM-focused ML coding interviews.
