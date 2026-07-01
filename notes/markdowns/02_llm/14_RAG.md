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

## Hybrid Search (BM25 + Dense)

### Why Hybrid?

- **Dense (embedding) search**: good at semantic similarity ("happy" ≈ "joyful"), bad at exact keyword matching.
- **Sparse (BM25) search**: good at exact keyword matching (product codes, names, rare terms), bad at semantics.

Hybrid search combines both for the best of both worlds.

### BM25 (Best Matching 25)

```
score(D, Q) = Σ IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
```

Where:
- `f(q_i, D)` = frequency of query term in document
- `IDF(q_i)` = inverse document frequency
- `|D|` = document length, `avgdl` = average document length
- `k1` (typically 1.2), `b` (typically 0.75) are tuning parameters

### Reciprocal Rank Fusion (RRF)

Combine ranked lists from multiple retrieval methods:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where `rank_i(d)` is the rank of document `d` in retrieval method `i`, and `k` is typically 60.

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

### When to Use Hybrid

| Scenario | Best Method |
|---|---|
| Semantic questions ("how to handle errors") | Dense |
| Exact match (product codes, names) | Sparse (BM25) |
| Mixed queries | Hybrid |
| Multi-language | Dense (cross-lingual embeddings) |
| Legal/medical (precise terminology) | Hybrid |

---

## Reranking Deep Dive

### Why Reranking Is the Highest-ROI RAG Improvement

1. **Bi-encoder (retrieval)**: fast, approximate, embeds query and doc separately. Good for recall, bad for precision.
2. **Cross-encoder (reranker)**: slow, accurate, processes (query, doc) jointly. Good for precision.

The two-stage pipeline: **retrieve many (bi-encoder) → rerank few (cross-encoder)**.

### Cross-Encoder Architecture

```
Input: [CLS] query [SEP] document [SEP]
       ↓
       BERT/Transformer
       ↓
       [CLS] hidden state
       ↓
       Linear layer → relevance score (scalar)
```

The cross-encoder sees both query and document **together**, enabling it to model fine-grained interactions that bi-encoders miss.

### Performance Impact

| Stage | Method | Recall@100 | Precision@5 |
|---|---|---|---|
| Retrieval only | Bi-encoder | 85% | 40% |
| + Reranking | Bi-encoder + Cross-encoder | 85% | 70% |

Reranking doesn't improve recall (you can't retrieve what you missed), but it dramatically improves **precision** — the top results are much more relevant.

### Cost Analysis

```
Bi-encoder retrieval: O(N * D) for N docs — fast, ANN-accelerated
Cross-encoder reranking: O(K * L) for K docs of length L — slow but K is small (10-100)
```

### Implementation

```python
def retrieve_and_rerank(query, doc_store, embedder, reranker, k_retrieve=50, k_final=5):
    # Stage 1: Fast bi-encoder retrieval
    query_emb = embedder.encode(query)
    top_k_idx = doc_store.search(query_emb, k=k_retrieve)  # fast ANN

    # Stage 2: Slow cross-encoder reranking
    candidates = [doc_store.get_text(i) for i in top_k_idx]
    scores = [reranker.score(query, doc) for doc in candidates]  # cross-encoder

    # Sort by reranker score
    reranked = np.argsort(scores)[::-1][:k_final]
    return [top_k_idx[i] for i in reranked]
```

### Reranker Models

| Model | Type | Speed |
|---|---|---|
| Cohere Rerank | API | Fast |
| bge-reranker-large | Open-source cross-encoder | Medium |
| ColBERT | Late interaction (efficient cross-encoder) | Fast |
| RankGPT | LLM as reranker | Slow but powerful |

---

## Query Rewriting

### Why Rewrite Queries?

User queries are often ambiguous, incomplete, or poorly phrased for retrieval.

### Techniques

#### 1. Query Expansion

Add synonyms or related terms:

```
Original: "laptop battery life"
Expanded: "laptop battery life notebook battery duration power"
```

#### 2. HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, then embed it for retrieval:

```
Query: "What causes inflation?"
→ LLM generates hypothetical answer: "Inflation is caused by..."
→ Embed the hypothetical answer (not the query) for retrieval
```

The hypothetical answer is closer in embedding space to real documents than the short query.

#### 3. Multi-Query Generation

Generate multiple reformulations and retrieve for all:

```python
def multi_query_retrieve(query, llm, retriever, n_queries=3):
    reformulations = llm.generate(f"Rewrite this query {n_queries} ways: {query}")
    all_results = []
    for q in reformulations:
        all_results.extend(retriever.retrieve(q, k=5))
    # Deduplicate and rerank
    return deduplicate(all_results)
```

#### 4. Step-Back Prompting

For complex queries, generate a broader "step-back" query:

```
Original: "Why did Tesla's stock drop in Q3 2024?"
Step-back: "What factors affect Tesla's stock price?"
```

Retrieve for both the original and step-back query.

---

## Advanced Chunking Strategies

### Fixed-Size Chunking (Baseline)

```
Document: 10,000 tokens
Chunk size: 512 tokens
Overlap: 50 tokens
→ ~22 chunks
```

### Semantic Chunking

Split at natural boundaries (paragraphs, sections, sentences) instead of fixed sizes:

```python
def semantic_chunk(text, min_chunk=200, max_chunk=1000):
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_chunk:
            if current:
                chunks.append(current)
            current = para
        else:
            current += "\n\n" + para
    if current:
        chunks.append(current)
    return chunks
```

### Document-Aware Chunking

- **Markdown/HTML**: split by headers (`#`, `##`).
- **Code**: split by functions/classes.
- **PDF**: split by pages or sections.
- **Tables**: keep as single chunks (don't split mid-table).

### Parent-Child Chunking

```
Parent chunk: 2000 tokens (for context)
Child chunks: 200 tokens each (for retrieval)

Retrieve using child chunks → return parent chunk as context
```

This gives precise retrieval (small chunks) with rich context (large chunks).

### Late Chunking

Embed the full document first, then chunk the embeddings:

```
1. Pass full document through embedding model
2. Get token-level embeddings
3. Average pool token embeddings per chunk
```

Preserves cross-chunk context that's lost with independent chunk embedding.

---

## RAGAS (RAG Assessment Framework)

### Key Metrics

| Metric | What It Measures | How |
|---|---|---|
| Faithfulness | Is the answer grounded in retrieved context? | Check every claim in answer vs context |
| Answer Relevance | Does the answer address the question? | LLM judges relevance |
| Context Precision | Is the retrieved context relevant? | Check if context contains answer |
| Context Recall | Did retrieval find the right info? | Check if answer info is in context |

### Faithfulness Evaluation

```
1. Extract all claims from the generated answer
2. For each claim, check if it's supported by the retrieved context
3. Faithfulness = (supported claims) / (total claims)
```

This detects hallucinations — claims not supported by retrieved context.

### Why RAGAS Matters for L5

- Perplexity and BLEU don't work for RAG evaluation.
- You need to evaluate **retrieval quality** and **generation quality** separately.
- RAGAS gives a framework for systematic RAG evaluation.
- In production, track RAGAS metrics over time to detect degradation.

---

## Multi-Turn RAG

### The Problem

In a conversation, the user's follow-up question may reference previous context:

```
Turn 1: "What is RAG?"
Turn 2: "How does it compare to fine-tuning?"
```

"it" refers to RAG, but the retriever doesn't know that.

### Solutions

#### Query Rewriting with Context

```python
def rewrite_query(conversation_history, current_query, llm):
    prompt = f"""
    Conversation: {conversation_history}
    Current query: {current_query}
    Rewrite the current query to be self-contained:
    """
    return llm.generate(prompt)
    # "How does it compare to fine-tuning?" → "How does RAG compare to fine-tuning?"
```

#### Conversation Memory + Hybrid Retrieval

```
1. Rewrite query using conversation history
2. Retrieve using rewritten query
3. Also retrieve using original query (for fallback)
4. Merge and rerank results
```

---

## L5 Interview Q&A

### Q: "Design a RAG system for a company's internal knowledge base with 10M documents."

**Architecture:**
1. **Ingestion**: document parsing → semantic chunking → embedding (e.g., `bge-large-en`) → HNSW index.
2. **Retrieval**: hybrid search (BM25 + dense) → retrieve top 50 → cross-encoder rerank → top 5.
3. **Generation**: context-augmented prompt → LLM with citation instructions.
4. **Evaluation**: RAGAS metrics (faithfulness, answer relevance) in production monitoring.
5. **Infrastructure**: FAISS/HNSW for ANN, Elasticsearch for BM25, Redis for query cache.
6. **Updates**: incremental indexing for new documents, nightly full re-index for deleted/modified.

**Key decisions:**
- Chunk size: 512 tokens with 10% overlap (tune via ablation on Recall@K).
- Embedding model: `bge-large-en-v1.5` (1024 dim, good multilingual support).
- Reranker: `bge-reranker-large` (cross-encoder, top 50 → top 5).
- LLM: GPT-4 or Claude for generation, with structured citation output.

### Q: "How would you handle a RAG system where retrieval quality is poor?"

1. **Diagnose**: measure Recall@K and Precision@K on a labeled eval set.
2. **Improve embeddings**: try a better embedding model (e.g., `bge-large` vs `miniLM`).
3. **Hybrid search**: add BM25 for keyword matching.
4. **Query rewriting**: use HyDE or multi-query to improve recall.
5. **Reranking**: add a cross-encoder reranker (highest ROI).
6. **Chunking**: tune chunk size and overlap; try semantic or parent-child chunking.
7. **Increase K**: retrieve more candidates, then rerank harder.

### Q: "When would you use RAG vs long context vs fine-tuning?"

| Method | When to Use | Cost | Freshness |
|---|---|---|---|
| RAG | Large, changing knowledge base | Low (no training) | Real-time |
| Long context | Small, fixed document set | Medium (longer prompts) | Real-time |
| Fine-tuning | Style/format/domain adaptation | High (training) | Stale (needs retraining) |

**Rule of thumb**: RAG for knowledge, fine-tuning for behavior, long context for single-document analysis.

### Q: "How would you handle conflicting information in retrieved documents?"

1. **Reranking**: rank by source authority (official docs > wiki > forum).
2. **Recency**: prefer recent documents for time-sensitive info.
3. **Prompt design**: instruct LLM to note conflicts ("Sources disagree: A says X, B says Y").
4. **Citation**: always cite sources so users can verify.
5. **Metadata filtering**: filter by source reliability before retrieval.

### Q: "How do you evaluate RAG in production?"

1. **Online metrics**: track faithfulness (RAGAS), user feedback (thumbs up/down), citation click-through.
2. **Offline eval set**: maintain a golden set of (query, expected answer, relevant docs) — run weekly.
3. **Retrieval metrics**: Recall@K, MRR on the eval set.
4. **Generation metrics**: faithfulness, answer relevance, hallucination rate.
5. **A/B testing**: compare embedding models, chunk sizes, rerankers.
6. **Monitoring**: alert on faithfulness drops, retrieval latency spikes, empty result rates.

---

## Interview Sound Bites

- RAG grounds LLM generation in external facts — retrieval quality dominates final answer quality.
- **Hybrid search** (BM25 + dense + RRF) combines keyword matching with semantic understanding.
- **Reranking** with a cross-encoder is the highest-ROI improvement: retrieve 50, rerank to 5.
- **Query rewriting** (HyDE, multi-query, step-back) improves recall for ambiguous queries.
- **RAGAS** evaluates RAG systems: faithfulness (hallucination detection), answer relevance, context precision/recall.
- **Chunking strategy** matters: semantic > fixed-size; parent-child for precision + context.
- **Multi-turn RAG**: rewrite queries using conversation history before retrieval.
- ANN (HNSW/FAISS) trades small accuracy for large latency improvement at scale.
- RAG for knowledge, fine-tuning for behavior, long context for single-document analysis.
