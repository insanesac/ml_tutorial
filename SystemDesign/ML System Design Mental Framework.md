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
