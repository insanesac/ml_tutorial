# ML System Design Mental Framework (LLM Focused)

A structured approach to ML system design interviews, with emphasis on LLM-specific patterns.

---

## Universal Interview Answer Structure

```
Requirements → Metrics → Pattern Identification → Data → Features / Context
→ Model Choice → Serving Architecture → Scaling → Evaluation → Monitoring
→ Ethics → Tradeoffs
```

**Always communicate assumptions and tradeoffs.**

---

## 1. Clarify Requirements

### Functional Requirements

Ask:
- What are we building?
- What are the core user actions?
- What is the input and output?
- Is this real-time or batch?

### Non-Functional Requirements

Ask:
- **Scale:** How many users? QPS? Data volume?
- **Latency:** What is acceptable? p50? p99?
- **Availability:** What uptime is needed?
- **Cost constraints:** Budget per request? Per user?
- **Privacy / Security:** What data is sensitive? Regulatory requirements?

### Summarize Scope

> "Let me confirm my understanding — we are building [X] for [Y] users, with [Z] latency requirement, prioritizing [NFR]."

This shows structured thinking and gives the interviewer a chance to correct assumptions early.

---

## 2. Define Success Metrics

### Three Levels of Metrics

#### Business Metrics
What leadership cares about:
- User satisfaction
- Retention
- Engagement
- Acceptance rate (for suggestions)
- Task completion rate

#### System Metrics
What engineers optimize:
- Latency (p50, p99)
- Throughput (QPS)
- Cost per request
- Availability (uptime %)

#### Model Metrics
What ML engineers measure:
- Accuracy / Precision / Recall
- **Hallucination Rate** (LLM-specific)
- **Groundedness** (is the answer based on retrieved context?)
- Recall@K (retrieval quality)
- NDCG (ranking quality)

### Why All Three Matter

Optimizing only model metrics can hurt business metrics. A 99% accurate model that costs $10 per request and takes 10 seconds is useless if the business needs $0.01 cost and 1-second latency.

---

## 3. Identify System Pattern

### Pattern A: Pure Generation

**Examples:** Creative Writing Assistant, Email Writer, Marketing Content Generator

```
User → Prompt Builder → LLM → Response
```

**Key considerations:**
- Prompt engineering is critical
- No retrieval needed
- Latency dominated by LLM inference
- Hallucination risk is high (no grounding)

---

### Pattern B: RAG (Retrieval-Augmented Generation)

**Examples:** Enterprise Assistant, Document Q&A, Research Assistant, Legal Assistant

```
User → Retriever → Vector DB → Prompt Builder → LLM → Response
```

**Key considerations:**
- Retrieval quality directly impacts answer quality
- Chunking strategy matters
- Embedding model choice
- Vector DB selection (HNSW, IVF, brute-force)
- Groundedness is measurable
- Hallucination can be reduced by grounding

---

### Pattern C: Coding Assistant

**Examples:** GitHub Copilot, Code Review Assistant

```
IDE Context → Code Retrieval → Prompt Builder → LLM → Suggestion
```

**Key considerations:**
- Context window management (open files, imports, repo structure)
- Latency is critical (suggestions must appear as you type)
- Code-specific retrieval (semantic + syntactic)
- Acceptance rate is the key product metric

---

### Pattern D: Agent

**Examples:** Research Agent, Travel Planner, Multi-Agent Assistant

```
User → Planner → Tool Calls → LLM → Response
```

**Key considerations:**
- Tool selection and orchestration
- Multi-step reasoning
- Error handling for tool failures
- Latency is high (multiple LLM calls)
- Cost per request is high
- Evaluation is hard (multi-step correctness)

---

### Pattern E: Recommendation / Ranking

**Examples:** YouTube, Netflix, Product Recommendations

```
User → Candidate Generation → Ranking Model → Results
```

**Key considerations:**
- Two-stage: candidate generation (fast, broad) → ranking (slower, precise)
- Feature engineering is critical
- Recall@K and NDCG are key metrics
- Cold start problem for new users/items
- Online learning / model freshness

---

### Pattern F: Classification / Prediction

**Examples:** Fraud Detection, Churn Prediction, Spam Detection

```
Features → Model → Prediction
```

**Key considerations:**
- Feature pipeline is the bulk of the work
- Model interpretability may matter (regulatory)
- Real-time scoring vs batch scoring
- Class imbalance handling
- Precision-recall tradeoff (cost of false positive vs false negative)

---

## 4. Data Ingestion & Preprocessing

### Questions to Ask

- Where does data come from?
- How is it collected?
- How is it cleaned?
- Any augmentation needed?
- Any filtering needed?

### Common Data Sources

| Pattern | Data Sources |
|---|---|
| RAG | Documents, PDFs, wikis, knowledge bases |
| Coding Assistant | Repositories, code files, documentation |
| Recommendation | User interactions, clicks, views, purchases |
| Classification | Logs, user events, transaction records |
| Agent | Tool outputs, API responses, web pages |

### Preprocessing Steps

1. **Collection:** Batch import, streaming ingestion, API polling
2. **Cleaning:** Remove duplicates, fix encoding, handle missing data
3. **Chunking** (for RAG): Split documents into retrievable units
4. **Embedding:** Convert text/code into vector representations
5. **Indexing:** Store vectors in vector DB for fast retrieval

---

## 5. Features / Context

### The Key Question

> "What information does the model need to do its job?"

### Context by Pattern

| Pattern | Context Needed |
|---|---|
| **ChatGPT** | Conversation history, memory, system prompt, retrieved documents |
| **Copilot** | Current file, imports, open tabs, repo structure, cursor position |
| **Creative Writing** | Genre, tone, audience, style examples |
| **RAG** | Retrieved chunks, metadata, user query, conversation history |
| **Agent** | Tool descriptions, previous tool outputs, user goal, plan state |
| **Recommendation** | User features, item features, interaction history, context (time, device) |

### Context Window Management

For LLM systems, context window is a scarce resource:

- **Prioritize:** Most relevant context first
- **Truncate:** Drop old conversation turns
- **Compress:** Summarize long documents before including
- **Rerank:** Use a smaller model to rerank retrieved chunks before feeding to LLM

---

## 6. Model Selection

### Key Questions

- LLM or traditional ML?
- Fine-tuning needed?
- RAG needed?
- Tool use needed?
- Small model vs large model?

### The Quality-Latency-Cost Triangle

```
        Quality
           /\
          /  \
         /    \
        /      \
       /________\
  Latency      Cost
```

You can optimize for at most two of three.

### Decision Framework

| Need | Model Choice |
|---|---|
| High quality, cost OK | Large model (GPT-4, Claude, Gemini Ultra) |
| Balanced | Medium model (Llama 70B, Mistral) |
| Low latency, low cost | Small model (Llama 8B, Gemma 2B) |
| Domain-specific accuracy | Fine-tuned smaller model |
| Grounded answers | RAG + any model |
| Multi-step reasoning | Large model + tool use |

### Fine-Tuning vs RAG vs Prompt Engineering

| Approach | When to Use | Cost | Latency Impact |
|---|---|---|---|
| **Prompt Engineering** | Quick iteration, general tasks | Low | None |
| **RAG** | Knowledge-intensive, dynamic data | Medium | Retrieval latency |
| **Fine-tuning** | Domain-specific style/format, consistent behavior | High | None (same inference) |
| **All three** | Production-grade domain system | High | Combined |

---

## 7. Serving Architecture

### Typical Flow

```
User → API → Rate Limiter → Cache → DB → Build Context → Queue
→ Inference Service → Model → Response → DB → User
```

### Common Components

| Component | Purpose |
|---|---|
| **Rate Limiter** | Prevent abuse, control cost |
| **Cache** | Cache common queries / responses |
| **Database** | Store user data, conversation history |
| **Queue** | Buffer requests during traffic spikes |
| **Load Balancer** | Distribute across inference servers |
| **Inference Service** | Run model inference (GPU servers) |

### LLM-Specific Serving Concerns

- **GPU allocation:** Models need GPU memory. Plan for model size + KV cache.
- **Batching:** Group requests for efficient GPU utilization
- **Streaming:** Stream tokens to user for perceived low latency
- **Model versioning:** Blue-green deployments for model updates
- **Fallback:** If large model is overloaded, fall back to smaller model

---

## 8. Scaling & Optimization

### Latency Optimization

| Technique | How It Helps |
|---|---|
| **Streaming** | User sees first token in ~200ms instead of waiting for full response |
| **Caching** | Common queries return instantly from cache |
| **Prompt compression** | Shorter prompts = faster inference |
| **Speculative decoding** | Small model drafts, large model verifies |
| **KV cache** | Avoid recomputing past tokens during generation |

### Throughput Optimization

| Technique | How It Helps |
|---|---|
| **Batching** | Process multiple requests in one forward pass |
| **Autoscaling** | Add GPU servers during peak, remove during low traffic |
| **Queueing** | Buffer requests, process in batches |
| **Continuous batching** | Dynamically add/remove requests from active batch |

### Cost Optimization

| Technique | How It Helps |
|---|---|
| **Smaller models** | Lower GPU cost per request |
| **Quantization** | Reduce model precision (fp16 → int8 → int4) |
| **Caching** | Avoid redundant inference |
| **Distillation** | Train smaller model to mimic larger model |
| **Route by complexity** | Easy queries → small model, hard queries → large model |

### Reliability

| Technique | How It Helps |
|---|---|
| **Fallback models** | If primary model fails, use backup |
| **Retries** | Retry transient failures with backoff |
| **Circuit breakers** | Stop sending traffic to failing service |
| **Graceful degradation** | Return cached or simplified response during outages |

---

## 9. Evaluation

### Offline Evaluation

| Metric | What It Measures |
|---|---|
| Accuracy | Overall correctness |
| Precision | Of positive predictions, how many are correct |
| Recall | Of actual positives, how many did we find |
| F1 | Harmonic mean of precision and recall |
| Recall@K | Of top-K results, how many relevant items are found |
| NDCG | Ranking quality (normalized discounted cumulative gain) |
| BLEU / ROUGE | Text generation quality (n-gram overlap) |
| Perplexity | Language model quality (lower is better) |

### Online Evaluation

| Metric | What It Measures |
|---|---|
| CTR | Click-through rate (recommendations) |
| Acceptance Rate | How often users accept suggestions (coding assistant) |
| User Satisfaction | Survey scores, thumbs up/down |
| Retention | Do users come back? |
| Task Completion | Did the user complete their task? |

### A/B Testing

Test variations to find the best configuration:
- **Prompt versions:** Different system prompts
- **Model versions:** Different model sizes or fine-tunes
- **Retrieval strategies:** Different chunk sizes, embedding models, reranking

---

## 10. Monitoring

### System Monitoring

| Metric | Alert Threshold |
|---|---|
| Latency (p99) | > target latency |
| Error rate | > 1% of requests |
| Throughput | Unexpected drop |
| GPU utilization | > 90% (scale needed) or < 20% (scale down) |

### Model Monitoring

| Metric | Why It Matters |
|---|---|
| **Hallucinations** | Model generating false information |
| **Drift** | Input distribution changing over time |
| **Quality** | Output quality degrading (user feedback) |

### Business Monitoring

| Metric | Why It Matters |
|---|---|
| **Engagement** | Are users using the feature? |
| **Satisfaction** | Are users happy with results? |
| **Adoption** | Is usage growing? |

---

## 11. Safety / Ethics

### Considerations

| Concern | Mitigation |
|---|---|
| **Bias** | Diverse training data, bias audits, fairness metrics |
| **Privacy** | PII detection, data retention policies, user consent |
| **Toxicity** | Content filters, safety classifiers, red-teaming |
| **Hallucinations** | Grounding (RAG), confidence thresholds, human review |
| **Copyright** | Training data provenance, output filtering |
| **Data leakage** | Prompt injection defenses, output sanitization |

### LLM-Specific Safety

- **Prompt injection:** User input tries to override system instructions
- **Jailbreaking:** Users attempt to bypass safety guardrails
- **Data exfiltration:** Model reveals training data or system prompts
- **Model inversion:** Attackers reconstruct training data from outputs

---

## Quick Reference: The 11-Step Framework

```
 1. Requirements     → What are we building? For whom? What NFRs?
 2. Metrics          → How do we know it's working? (Business + System + Model)
 3. Pattern          → Which system pattern? (Generation, RAG, Agent, etc.)
 4. Data             → Where does data come from? How is it processed?
 5. Features/Context → What does the model need to see?
 6. Model Choice     → Which model? Fine-tune? RAG? Tradeoffs?
 7. Serving          → How do we serve it? (API, cache, queue, GPU)
 8. Scaling          → How do we handle growth? (Batching, autoscaling)
 9. Evaluation       → How do we measure quality? (Offline + Online)
10. Monitoring       → How do we detect problems? (System + Model + Business)
11. Ethics           → What are the risks? (Bias, privacy, hallucinations)
```

**Always end with tradeoffs.** Every choice has a cost. State it explicitly.

---

## 12. LLM Serving Architecture Deep Dive

### GPU Capacity Planning

```
GPU Memory Budget = Model Weights + KV Cache + Activations + Overhead

Example: LLaMA-2 70B on 8x A100 80GB
  Weights (FP16):     140 GB → 17.5 GB/GPU (tensor parallelism)
  KV Cache (batch=32, ctx=4096): ~32 GB → 4 GB/GPU
  Activations:        ~8 GB → 1 GB/GPU
  Overhead:           ~2 GB/GPU
  Total per GPU:      ~24.5 GB (fits in 80 GB)
```

### Serving Stack Components

```
User Request
    ↓
API Gateway (auth, rate limiting, routing)
    ↓
Request Queue (Kafka/SQS for async, or in-memory for sync)
    ↓
Scheduler (continuous batching, priority, admission control)
    ↓
Inference Engine (vLLM, TGI, TensorRT-LLM)
    ↓
Model (tensor-parallel across GPUs)
    ↓
Response Stream (SSE/WebSocket for token streaming)
    ↓
User
```

### Key Serving Decisions

| Decision | Options | Tradeoff |
|---|---|---|
| **Serving framework** | vLLM, TGI, TensorRT-LLM, Triton | Throughput vs flexibility vs vendor lock-in |
| **Parallelism** | Tensor parallel (TP), Pipeline parallel (PP), Data parallel (DP) | TP: communication heavy; PP: bubble overhead; DP: simplest |
| **Batching** | Static, dynamic, continuous | Static: simple but wasteful; Continuous: optimal GPU util |
| **Quantization** | FP16, INT8, INT4 | Quality vs memory vs speed |
| **Caching** | Prefix cache, semantic cache | Hit rate vs freshness |

### Model Routing (Cascading)

```
User Query → Classifier → Simple? → Small Model (8B)
                       → Medium? → Medium Model (70B)
                       → Complex? → Large Model (GPT-4)
```

```python
def route_request(query, classifier):
    complexity = classifier.score(query)
    if complexity < 0.3:
        return small_model.generate(query)    # 8B, fast, cheap
    elif complexity < 0.7:
        return medium_model.generate(query)   # 70B, balanced
    else:
        return large_model.generate(query)    # GPT-4, slow, expensive
```

**Benefits**: 70-80% of queries are simple → route to small model → 10x cost savings with minimal quality loss.

### Continuous Batching (L5 Critical)

Traditional batching waits for all requests in a batch to finish → GPU idle time. Continuous batching:

```
Time:  t1    t2    t3    t4    t5
Req A: [gen] [gen] [gen] [done]
Req B:       [gen] [gen] [gen] [gen] [done]
Req C:             [gen] [gen] [gen] [done]
Req D:                   [gen] [gen] [gen] [done]

GPU never idle — new requests join as old ones finish
```

This is what vLLM and TGI implement. It's the single biggest throughput improvement for LLM serving.

### Streaming Architecture

```
Client ← SSE ← API ← Token Stream ← Inference Engine

Token 1: "The" → sent immediately (~200ms TTFT)
Token 2: "answer" → sent as generated (~20ms TPOT)
Token 3: "is" → sent as generated
...
```

**TTFT (Time To First Token)**: dominated by prefill (processing the prompt).
**TPOT (Time Per Output Token)**: dominated by decode (one forward pass per token).

### Multi-Model Serving

```
GPU 0-3: Model A (70B, TP=4)
GPU 4-5: Model B (8B, TP=2)
GPU 6-7: Model C (8B, TP=2) — for fallback/overflow
```

Considerations:
- **Model placement**: which GPUs hold which models.
- **Request routing**: send to the right model's GPU group.
- **Hot-swapping**: load/unload models without downtime.
- **Multi-LoRA**: one base model + multiple adapters (see LoRA notes).

---

## 13. RAG System Design (End-to-End)

### Architecture

```
User Query
    ↓
Query Rewriter (LLM: HyDE, multi-query, step-back)
    ↓
Hybrid Retriever
    ├── BM25 (Elasticsearch) → top 50
    └── Dense (FAISS/HNSW)   → top 50
    ↓
Reciprocal Rank Fusion (RRF) → top 50 merged
    ↓
Cross-Encoder Reranker → top 5
    ↓
Context Builder (chunk selection, token budget management)
    ↓
LLM Generation (with citation instructions)
    ↓
Response + Citations → User
```

### Key Design Decisions

| Component | Decision | Impact |
|---|---|---|
| **Embedding model** | bge-large-en (1024 dim) vs OpenAI ada (1536 dim) | Quality vs cost |
| **Chunk size** | 512 tokens, 10% overlap | Precision vs context |
| **Vector DB** | FAISS (in-memory) vs Pinecone (managed) vs pgvector (Postgres) | Latency vs ops vs integration |
| **Reranker** | bge-reranker-large vs Cohere Rerank | Quality vs cost |
| **Index update** | Real-time incremental vs nightly batch | Freshness vs complexity |

### Indexing Pipeline

```
Documents → Parser (PDF/HTML/Markdown) → Semantic Chunker
    ↓
Embedding Model → Vectors
    ↓
Vector DB (HNSW index) + Metadata Store (Postgres)
    ↓
Incremental updates for new/modified documents
Nightly full re-index for deletions
```

### Token Budget Management

```
Total context budget: 8192 tokens
  System prompt:     500 tokens
  Retrieved context: 5000 tokens (5 chunks × 1000 tokens)
  Conversation:      1000 tokens
  Response buffer:   1692 tokens

If retrieved context exceeds budget → truncate or summarize
```

---

## 14. Agent System Design

### Architecture

```
User Goal
    ↓
Planner LLM: decompose goal into steps
    ↓
Step 1: Select tool → Execute → Observe result
Step 2: Select tool → Execute → Observe result
...
Step N: Synthesize final answer
    ↓
Response → User
```

### Tool Selection

```
Available tools:
  - search_web(query) → search results
  - calculator(expression) → numeric result
  - code_executor(code) → execution output
  - database_query(sql) → query results

Planner prompt: "Given the goal and available tools, which tool should I use next?"
```

### Key Challenges

| Challenge | Solution |
|---|---|
| **Tool failure** | Retry with backoff, fallback to alternative tool |
| **Infinite loops** | Max steps limit, cycle detection |
| **Cost control** | Budget per request (max LLM calls, max tool calls) |
| **Latency** | Parallel tool execution where possible |
| **Evaluation** | Step-by-step correctness checking |

### Multi-Agent Architecture

```
User → Orchestrator Agent
         ├── Research Agent (search + summarize)
         ├── Code Agent (write + execute code)
         └── Critic Agent (verify outputs)
```

Each agent has its own system prompt, tools, and evaluation criteria. The orchestrator coordinates and synthesizes.

---

## 15. L5 Interview Q&A

### Q: "Design an LLM serving system for 10M daily users."

**Scale estimation:**
- 10M users, ~5 queries/day = 50M queries/day
- ~580 queries/second average, ~3000 QPS peak (5x)
- Average response: 200 tokens, average prompt: 500 tokens

**Architecture:**
1. **API Gateway**: auth, rate limiting, request routing.
2. **Model selection**: route by complexity — 80% to 8B model, 15% to 70B, 5% to GPT-4.
3. **Serving**: vLLM with continuous batching, tensor parallelism (TP=4 for 70B, TP=1 for 8B).
4. **GPU fleet**: 8B model on 4 GPU groups (1 GPU each), 70B on 4 GPU groups (4 GPUs each).
5. **Caching**: prefix cache for shared system prompts, semantic cache for common queries.
6. **Queue**: Kafka for async requests, in-memory for sync streaming.
7. **Monitoring**: TTFT < 500ms, TPOT < 50ms, GPU utilization 60-80%.

**Cost estimate:**
- 80% queries on 8B: 40M queries × 700 tokens × $0.0001/1K tokens = $2,800/day
- 15% on 70B: 7.5M × 700 × $0.001/1K = $5,250/day
- 5% on GPT-4: 2.5M × 700 × $0.03/1K = $52,500/day
- Total: ~$60K/day → routing to small models saves ~$50K/day vs all-GPT-4.

### Q: "How would you reduce LLM serving costs by 10x?"

1. **Model routing**: 80% of queries to small model → 10x cheaper for those.
2. **Quantization**: INT4 for 70B → 4x less memory → more concurrent requests per GPU.
3. **Caching**: semantic cache for common queries → 20-30% hit rate = 20-30% fewer inference calls.
4. **Speculative decoding**: small model drafts, large model verifies → 2-3x throughput.
5. **Batching**: continuous batching → 3-5x GPU utilization improvement.
6. **Prompt compression**: reduce prompt length → less prefill compute.
7. **Off-peak scaling**: autoscale down during low traffic.

### Q: "How do you handle a traffic spike where QPS doubles suddenly?"

1. **Autoscaling**: pre-provisioned GPU pools that spin up in 2-5 minutes.
2. **Queue + backpressure**: buffer excess requests, degrade gracefully (shorter responses, smaller model).
3. **Model fallback**: if 70B is overloaded, route to 8B with a warning.
4. **Rate limiting**: throttle low-priority requests, prioritize paying users.
5. **Circuit breakers**: stop sending traffic to overloaded GPU groups.
6. **Pre-warming**: keep spare GPUs loaded with model weights (ready to serve).

### Q: "Design a RAG system that stays fresh as documents are updated."

1. **Incremental indexing**: new/modified documents → embed → add to vector DB in real-time.
2. **Versioning**: each chunk has a version timestamp; retrieval prefers recent versions.
3. **TTL on cached responses**: cached RAG answers expire after 1 hour (or on document update).
4. **Webhook integration**: document management system sends update events → trigger re-index.
5. **Nightly full re-index**: catch deletions and fix any incremental errors.
6. **Stale detection**: if retrieved chunks are >30 days old, flag to user ("Information may be outdated").

### Q: "How would you evaluate an LLM system in production?"

**Offline:**
- Benchmark suite: MMLU, GSM8K, HumanEval, domain-specific eval set.
- RAGAS: faithfulness, answer relevance, context precision/recall.
- Regression testing: ensure new model versions don't break existing capabilities.

**Online:**
- A/B testing: new model vs current model, measure user satisfaction.
- Human eval: sample 100 responses/day, human raters score quality.
- User feedback: thumbs up/down, detailed feedback forms.
- Business metrics: task completion rate, retention, support ticket deflection.

**Monitoring:**
- Quality drift: track faithfulness, hallucination rate over time.
- Safety: toxicity classifier, jailbreak detection, prompt injection attempts.
- Performance: TTFT, TPOT, error rate, GPU utilization.

---

## Quick Reference: LLM Serving Cheat Sheet

| Problem | Solution |
|---|---|
| High latency (TTFT) | Prefix caching, smaller model, prompt compression |
| High latency (TPOT) | Speculative decoding, quantization, smaller model |
| Low throughput | Continuous batching, tensor parallelism, larger batch |
| High cost | Model routing, caching, quantization, smaller model |
| OOM on GPU | Quantization, smaller batch, gradient checkpointing (training) |
| Hallucinations | RAG grounding, faithfulness eval, confidence thresholds |
| Stale knowledge | RAG with real-time indexing, webhook-triggered updates |
| Traffic spikes | Autoscaling, queueing, model fallback, rate limiting |
| Multi-tenant | Multi-LoRA serving, per-tenant adapters, isolated caches |
