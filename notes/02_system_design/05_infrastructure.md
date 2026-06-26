# System Design Foundations - Session 5

Built through discussion and exercises.

---

## 1. Single Point of Failure (SPOF)

### What Is a SPOF?

A **Single Point of Failure** is one component whose failure can take down the entire system.

```
Client → Load Balancer → [Server A]  ← if this is the only server, it's a SPOF
```

### Why SPOFs Are Dangerous

- No redundancy means no fallback
- One hardware failure = total outage
- One network glitch = all users affected

### How to Eliminate SPOFs

| Strategy | What It Does |
|---|---|
| **Redundancy** | Maintain backup instances of critical components |
| **Replication** | Multiple copies of data across machines |
| **Failover** | Automatic switch to backup when primary fails |

### Common SPOFs People Forget

- Load balancers (yes, they can be SPOFs too)
- Databases (single primary with no replicas)
- DNS servers
- Identity providers / auth services
- Cloud provider regions (single-region deployment)

---

## 2. Redundancy

### What Is Redundancy?

Maintain **backup instances** of critical components so that if one fails, another takes over.

### Types of Redundancy

| Type | Example |
|---|---|
| **Active-Active** | Two servers both serving traffic. If one dies, the other continues. |
| **Active-Passive** | One primary serves, one standby waits. Failover switches to standby. |
| **N+1** | N instances needed for load, +1 spare for failures. |

### Tradeoff

More redundancy = more cost and complexity. But less redundancy = more risk.

**Rule of thumb:** Redundancy should match the criticality of the component.

---

## 3. Fault Tolerance

### Definition

The system **continues operating** despite failures.

### Fault Tolerance vs High Availability

| | Fault Tolerance | High Availability |
|---|---|---|
| **Goal** | No downtime at all | Minimize downtime |
| **Cost** | Very expensive (full redundancy) | Moderate (some redundancy + fast recovery) |
| **Example** | Aircraft flight controls | Web application |

Most systems aim for **high availability**, not full fault tolerance. True fault tolerance is expensive.

---

## 4. Quorum

### What Is Quorum?

**Majority agreement** used for leadership and coordination.

If you have 5 servers, quorum = 3. Any decision agreed upon by 3+ servers is committed.

### Why Quorum Works

- Tolerates minority failures (2 of 5 can die)
- Prevents split brain (two majorities cannot coexist)
- Guarantees consistency (majority always overlaps)

### Quorum Math

| Cluster Size | Quorum | Max Failures Tolerated |
|---|---|---|
| 3 | 2 | 1 |
| 5 | 3 | 2 |
| 7 | 4 | 3 |

**Odd numbers are preferred** — even-sized clusters tolerate the same failures as one smaller, but with more overhead.

---

## 5. Randomized Election Timeouts

### The Problem

When multiple followers attempt to become leader simultaneously, they can **split the vote** — no one gets a majority.

### The Solution

**Randomized election timeouts:** Each follower waits a random duration before starting an election.

This makes it statistically unlikely that two followers start elections at the same time.

### Why It Works

- One follower times out first, starts election
- Others receive the election request and vote
- Split votes become rare

### Used In

- **Raft consensus** (core part of the protocol)
- Many leader election systems

---

## 6. Stable Leadership

### Why Stability Matters

Constant re-elections cause:
- Write interruptions
- Cluster instability
- User-visible latency spikes

### How to Maintain Stability

- **Longer election timeouts** on stable networks
- **Pre-vote protocol** (Raft optimization): check if others would vote before starting full election
- **Leader lease:** Leader keeps authority for a bounded time even if heartbeats are delayed

---

## 7. Load Balancer

### What It Does

Distributes traffic across multiple servers.

```
         → Server A
Client → Load Balancer → Server B
         → Server C
```

### Load Balancing Algorithms

| Algorithm | How It Works |
|---|---|
| **Round Robin** | Rotate through servers sequentially |
| **Least Connections** | Send to server with fewest active connections |
| **IP Hash** | Same client always goes to same server (sticky) |
| **Weighted** | Distribute based on server capacity |

### Load Balancer SPOF

Load balancers themselves require redundancy!

If your load balancer dies, all traffic stops — even if all backend servers are healthy.

**Solution:** Deploy load balancers in active-active or active-passive pairs.

---

## 8. Cache

### What It Does

Store frequently requested data/results for **faster access**.

### Cache Hit vs Miss

| Event | What Happens |
|---|---|
| **Cache hit** | Request served from cache. Fast. No backend call. |
| **Cache miss** | Request goes to backend. Result stored in cache for next time. |

### Cache Invalidation

The challenge of keeping cached data **fresh**.

| Strategy | Description |
|---|---|
| **TTL (Time to Live)** | Cache entry expires after a set duration |
| **Write-through** | Update cache immediately when source data changes |
| **Write-behind** | Update cache first, persist to DB later |
| **Explicit invalidation** | Manually evict cache entries when data changes |

### Interview Trap

> "Cache invalidation is one of the hardest problems in computer science."

Know which strategy you are using and **why**.

---

## 9. Database

### What It Does

**Persistent storage** that survives restarts and power failures.

### Common Database Types

| Type | Example | Best For |
|---|---|---|
| **Relational (SQL)** | PostgreSQL, MySQL | Structured data, transactions, strong consistency |
| **Document (NoSQL)** | MongoDB, DynamoDB | Flexible schemas, horizontal scaling |
| **Key-Value** | Redis, Memcached | Fast lookups, caching |
| **Column-family** | Cassandra, HBase | Wide-column data, write-heavy workloads |
| **Graph** | Neo4j | Relationships, social networks |
| **Time-series** | InfluxDB, TimescaleDB | Metrics, monitoring data |

---

## 10. Replication

### What It Does

**Multiple copies** of the same data for availability and read scaling.

### Replication Patterns

| Pattern | How It Works |
|---|---|
| **Primary-Replica** | Primary handles writes, replicas handle reads |
| **Multi-Primary** | Any node can accept writes (conflict resolution needed) |
| **Sync Replication** | Write acknowledged only after all replicas confirm |
| **Async Replication** | Write acknowledged immediately, replicas catch up later |

### Tradeoff

- **Sync:** Safe but slow
- **Async:** Fast but risk of data loss on failure

---

## 11. Sharding

### What It Does

**Split data across machines** for storage and write scaling.

### Sharding Strategies

| Strategy | How It Works |
|---|---|
| **Range-based** | Shard by key ranges (e.g., A-M on shard 1, N-Z on shard 2) |
| **Hash-based** | Shard by hash of key (even distribution) |
| **Geo-based** | Shard by geography (EU users on EU servers) |

### Challenges

- **Hot shards:** Uneven distribution
- **Cross-shard queries:** Joins across shards are expensive
- **Resharding:** Adding shards requires data migration

---

## 12. Message Queue

### What It Does

**Buffers work** between producers and consumers.

```
Producer → [Queue] → Consumer
```

### Why Queues Help

- **Decouple producers from consumers:** Producers do not wait for consumers
- **Handle traffic spikes:** Queue absorbs burst, consumers process at their own pace
- **Retry on failure:** Failed messages can be reprocessed

### Queue Backlogs

When consumers are **slower than producers**, work accumulates in the queue.

**Symptoms:**
- Queue depth grows
- Latency increases (messages wait longer)
- Eventually, queue may overflow

**Solutions:**
- Add more consumers (scale out)
- Increase consumer throughput (batching, optimization)
- Backpressure (tell producers to slow down)

---

## 13. Search Index

### What It Does

**Precomputed structures** for fast search.

### Why Not Just Scan?

Scanning every document for a keyword is O(N). An inverted index makes it O(1) lookup + O(matches).

### Index Consistency

Indexes must reflect changes in source data.

| Problem | Solution |
|---|---|
| Document updated but index not refreshed | Reindex on write or periodic reindexing |
| Index and DB out of sync | Eventual consistency with reconciliation |

### Common Search Systems

- **Elasticsearch:** Full-text search, inverted index
- **Vector DB:** Similarity search, ANN index (HNSW, IVF)

---

## 14. CDN (Content Delivery Network)

### What It Does

Serve content from locations **close to users**.

```
User in India → CDN edge in Mumbai → Content
                    (instead of)
User in India → Origin server in US → Content (slow)
```

### Why CDNs Help

- **Lower latency:** Content served from nearby edge
- **Lower origin load:** Edge caches absorb most requests
- **Better availability:** Edge can serve cached content even if origin is down

### When to Use a CDN

- Static assets (images, CSS, JS)
- Video streaming
- Large file downloads
- API responses that are cacheable

---

## 15. Object Storage

### What It Does

Store **large files** such as videos, images, and PDFs.

### Examples

- **AWS S3**
- **Google Cloud Storage**
- **Azure Blob Storage**

### Why Not a Database?

Databases are optimized for structured queries, not large binary blobs. Object storage is:
- Cheaper per GB
- Designed for large files
- Horizontally scalable by default
- Often integrated with CDN for delivery

---

## Cheat Sheet

| Problem | Solution |
|---|---|
| Too much traffic | **Load Balancer** |
| Repeated expensive work | **Cache** |
| Persistent data | **Database** |
| Database failure | **Replication** |
| Database too large | **Sharding** |
| Traffic spikes | **Queue** |
| Fast search | **Search Index** |
| Large files | **Object Storage** |
| Global latency | **CDN** |
| Single point of failure | **Redundancy** |
| Coordination / consensus | **Quorum** |
| Leader election conflicts | **Randomized Timeouts** |

---

## Interview Checklist (Infrastructure Components)

When designing system infrastructure:

- [ ] Where are the SPOFs? How are they eliminated?
- [ ] Is traffic load balanced? How?
- [ ] What is cached? What is the invalidation strategy?
- [ ] What database type? Why?
- [ ] Is data replicated? Sync or async?
- [ ] Is data sharded? What sharding key?
- [ ] Are queues needed for traffic spikes?
- [ ] Is search needed? What index type?
- [ ] Are large files stored in object storage?
- [ ] Is a CDN needed for global users?

---

## L5 Additions: ML/LLM Infrastructure Components

### GPU Cluster Architecture

```
Request → Load Balancer → Inference Server (GPU)
                              ↓
                    Model loaded in GPU memory
                              ↓
                    Forward pass → Response
```

### GPU-Specific Infrastructure

| Component | Purpose | ML-Specific Concern |
|---|---|---|
| **GPU scheduler** | Allocate GPUs to jobs | GPU memory is the bottleneck, not just CPU |
| **Model registry** | Store/load model weights | Versioning, A/B deployment, weight distribution |
| **Feature store** | Serve ML features in real-time | Low-latency feature retrieval for online inference |
| **Vector DB** | Store/retrieve embeddings | HNSW/IVF index, ANN search, metadata filtering |
| **GPU monitoring** | Track GPU health | GPU temperature, memory usage, utilization, ECC errors |
| **Checkpoint storage** | Save training state | High-throughput write (TBs), distributed filesystem |

### Vector Database Selection

| Vector DB | Type | Best For | Scale |
|---|---|---|---|
| **FAISS** | In-memory library | Single-machine, low latency | ~100M vectors |
| **HNSW (hnswlib)** | In-memory graph | High recall, low latency | ~50M vectors |
| **Pinecone** | Managed service | No-ops, auto-scaling | Billions |
| **Weaviate** | Open-source server | Hybrid search (vector + keyword) | ~1B vectors |
| **pgvector** | Postgres extension | Small scale, SQL integration | ~10M vectors |
| **Milvus** | Distributed vector DB | Billion-scale, hybrid | Billions |
| **Qdrant** | Rust-based, fast | Performance-critical, filtering | ~1B vectors |

### Feature Store Architecture

```
Offline (training):
  Raw data → Feature Engineering → Feature Store (offline)
                                        ↓
Online (inference):
  Request → Feature Store (online, Redis) → Model → Response
```

**Consistency**: Online and offline features must match. If training uses "user_age_30d" but online serves "user_age_current", the model sees different distributions.

### Model Registry

```
Model Registry:
  model_id: "llama-2-70b-chat"
  version: "v1.3"
  weights: s3://models/llama-2-70b-chat/v1.3/weights.bin
  config: s3://models/llama-2-70b-chat/v1.3/config.json
  metrics: {faithfulness: 0.92, latency_p99: 450ms}
  status: "production" | "staging" | "archived"
```

### LLM-Specific Caching

| Cache Type | What It Caches | Hit Rate | Freshness |
|---|---|---|---|
| **Prefix cache** | KV cache for shared prompt prefix | High (shared system prompts) | Per session |
| **Semantic cache** | Responses for similar queries | 20-30% | TTL-based |
| **Exact cache** | Responses for identical queries | Low (rare exact repeats) | TTL-based |
| **Embedding cache** | Computed embeddings | High (documents don't change often) | On document update |

```python
def cached_generate(query, cache, model):
    # 1. Check exact cache
    if query in cache:
        return cache[query]
    
    # 2. Check semantic cache (embedding similarity)
    query_emb = embed(query)
    similar = cache.find_similar(query_emb, threshold=0.95)
    if similar:
        return similar.response
    
    # 3. Generate new response
    response = model.generate(query)
    
    # 4. Cache it
    cache.set(query, response, query_emb, ttl=3600)
    return response
```

### ML Monitoring Stack

```
System Metrics:
  GPU utilization, memory, temperature, ECC errors
  Network bandwidth (critical for TP/PP)
  Disk I/O (checkpoint loading, model weights)

Model Metrics:
  Inference latency (TTFT, TPOT)
  Output quality (faithfulness, hallucination rate)
  Request rate, batch size, queue depth

Business Metrics:
  User satisfaction (thumbs up/down)
  Task completion rate
  Cost per query
  Revenue impact
```

### L5 Interview Q&A

#### Q: "How do you choose between FAISS, Pinecone, and pgvector for a RAG system?"

| Factor | FAISS | Pinecone | pgvector |
|---|---|---|---|
| Scale | <100M vectors | Billions | <10M vectors |
| Latency | <1ms (in-memory) | ~10ms (network) | ~5ms (Postgres) |
| Ops | DIY (you manage) | Managed (no-ops) | Easy (if you have Postgres) |
| Cost | Free (self-hosted) | $$ (managed) | Free (if you have Postgres) |
| Filtering | Limited | Rich metadata filtering | Full SQL filtering |
| Best for | Prototype, low-latency | Production at scale | Small scale, SQL integration |

**Decision**: Start with FAISS for prototyping → move to Pinecone/Milvus for production at scale → use pgvector if you already have Postgres and scale is small.

#### Q: "Design a feature store for a real-time ML system."

```
Offline path (batch):
  Data warehouse → Spark/Beam feature pipeline → Offline feature store (S3/BigQuery)
  Used for: training data generation, batch scoring

Online path (real-time):
  User request → Online feature store (Redis/DynamoDB) → Model inference
  Used for: real-time scoring (<10ms retrieval)

Sync: Offline → Online sync pipeline (hourly or streaming)
```

**Key challenges**:
1. **Online/offline consistency**: same feature definition, same transformations.
2. **Freshness**: how quickly do new features appear online? (streaming vs batch)
3. **Point-in-time correctness**: training must use features as they were at event time, not current values.

#### Q: "How do you implement semantic caching for an LLM serving system?"

```
1. Compute embedding of incoming query
2. Search cache for similar queries (cosine similarity > 0.95)
3. If found: return cached response (with TTL check)
4. If not: generate response, cache with embedding + TTL

Cache structure:
  Key: query embedding (1024-dim vector)
  Value: response text + timestamp + metadata
  Index: HNSW for fast similarity search
  TTL: 1 hour (or invalidate on knowledge base update)

Hit rate: typically 20-30% for production LLM systems
Savings: 20-30% fewer inference calls → significant cost reduction
```

**Risks**:
- **Stale answers**: cached response may be outdated. Mitigate with TTL.
- **False positives**: semantically similar but functionally different queries. Mitigate with high threshold (0.95+).
- **Cache invalidation**: when knowledge base updates, invalidate relevant cache entries.

#### Q: "What infrastructure do you need to serve 10 different LLM models?"

```
Model fleet:
  - 2x large models (70B): 8 GPUs each (TP=4, 2 replicas)
  - 3x medium models (13B): 2 GPUs each (TP=1, 2 replicas each)
  - 5x small models (8B): 1 GPU each (2 replicas each)

Total GPUs: 16 + 12 + 10 = 38 GPUs

Components:
  - Model registry: track versions, weights, configs
  - GPU scheduler: allocate GPUs to models, handle failures
  - Load balancer: route to correct model + replica
  - Health checker: GPU temperature, memory, inference latency
  - Autoscaler: add/remove replicas based on load
  - Monitoring: per-model metrics (QPS, latency, quality)
  - Shared cache: prefix cache for common system prompts
```

#### Q: "How do you handle GPU OOM in production?"

1. **Prevention**: 
   - Set max batch size based on available memory.
   - Set max context length per request.
   - Monitor memory usage, alert at 80%.

2. **Detection**:
   - Catch OOM errors from inference engine.
   - Log the request that caused OOM (prompt length, batch size).

3. **Recovery**:
   - Drop the offending request, return error to user.
   - Reduce batch size temporarily.
   - If persistent: restart the inference server (reload model).

4. **Long-term**:
   - Quantize model (FP16 → INT8 → INT4).
   - Use GQA/MQA to reduce KV cache memory.
   - Add more GPUs (increase TP or add replicas).
   - Implement request admission control (reject very long prompts).
