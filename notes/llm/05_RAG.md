# RAG (Retrieval-Augmented Generation)

## What Is RAG?

RAG reduces hallucinations and gives an LLM access to external knowledge without retraining.

### Pipeline

```
User Query
    ↓
Query Embedding
    ↓
Vector Search (cosine similarity or ANN)
    ↓
Top-K Retrieval
    ↓
Context Augmentation (inject into prompt)
    ↓
LLM Generation
```

---

## Cosine Similarity

Measures angle between two vectors, ignoring magnitude:

```
cos(a, b) = dot(a, b) / (||a|| * ||b||)
```

| Value | Meaning |
|---|---|
| 1 | Same direction (identical) |
| 0 | Orthogonal (unrelated) |
| -1 | Opposite direction |

---

## Retrieval Implementation

```python
def retrieve(query_emb, doc_embs, k=5):
    # Normalize
    q = query_emb / np.linalg.norm(query_emb)
    D = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

    # Cosine similarity
    scores = D @ q

    # Top-K
    top_k_idx = np.argpartition(scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
    return top_k_idx
```

### Complexity

| Operation | Complexity |
|---|---|
| Brute-force similarity | O(N * D) |
| Argsort for top-K | O(N log N) |
| Argpartition for top-K | O(N) |

---

## Chunking

### Why Chunk?

Large documents dilute embeddings and hurt retrieval precision.

Chunking gives the model smaller, focused context windows to embed and retrieve.

### Chunk Size Tradeoff

| Small Chunks | Large Chunks |
|---|---|
| Better precision | More context per retrieved chunk |
| Less context | Worse precision |

### Chunk Overlap

Use 10–20% overlap between adjacent chunks to preserve context across boundaries.

---

## Failure Modes

### Retrieval Failure

Most common RAG failure mode.

The correct document exists in the corpus but is not retrieved in the top-K.

How to fix:
- Better embedding model
- Increase K
- Reranking (cross-encoder)
- Hybrid search (BM25 + dense)

### Hallucination

Even with good retrieval, LLM may:
- Ignore retrieved context
- Mix retrieved facts with model priors
- Confabulate on ambiguous context

Mitigations: grounding prompts, citation generation, faithfulness evaluation.

---

## ANN (Approximate Nearest Neighbor)

Brute-force search is `O(N * D)` — infeasible at 100M+ documents.

ANN trades a small accuracy loss for large speed gains.

### Common ANN Libraries

| Library | Algorithm |
|---|---|
| FAISS | IVF, HNSW, PQ |
| ScaNN | HNSW + quantization |
| Weaviate / Pinecone | Managed HNSW |

### HNSW Intuition

Hierarchical Navigable Small Worlds.

1. Start at a coarse upper layer.
2. Navigate greedily toward query.
3. Descend layers, refining the search.
4. Return nearest neighbors at the bottom layer.

---

## Vector Database

Stores three things:
1. Embeddings (for search)
2. Original text (for context injection)
3. Metadata (for filtering)

The LLM consumes **text**, not embeddings.

---

## Evaluation Metrics

| Metric | What It Measures |
|---|---|
| Recall@K | Fraction of relevant docs found in top-K |
| Precision@K | Fraction of top-K that are relevant |
| MRR | Mean Reciprocal Rank of first relevant result |

---

## Production Considerations

- Chunk size and overlap strategy
- Embedding model quality and dimension
- K selection (over-retrieve and rerank vs exact-K)
- Reranking with a cross-encoder
- Latency budget (ANN vs brute force tradeoff)
- Monitoring retrieval quality over time

---

## L5 Discussion Topics

| Question | Key Answer |
|---|---|
| Reduce hallucinations? | Better retrieval, grounding prompts, faithfulness eval |
| Improve Recall@K? | Better embedder, hybrid search, increase K + rerank |
| Handle 100M docs? | ANN (HNSW/FAISS), sharding, async indexing |
| Reduce retrieval latency? | ANN, caching hot queries, quantized embeddings |
| Choose chunk size? | Domain-specific tuning, ablation on Recall@K |

---

## Interview Sound Bites

- RAG grounds LLM generation in external facts.
- Retrieval quality dominates final answer quality.
- ANN improves latency at the cost of small accuracy loss.
- Chunking improves retrieval precision.
- Reranking is the cheapest high-impact improvement after baseline RAG.
