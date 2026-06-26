# System Design Foundations - Session 3

Built through discussion and exercises.

---

## 1. Why Estimate Scale?

### Scale Estimation Is Not About Math

The goal is not to produce exact numbers. The goal is to **understand how big the problem is** and whether a simple architecture is sufficient.

### What Scale Estimation Reveals

- Can a single machine handle this? Or do we need distributed systems?
- What is the dominant resource constraint? (CPU, memory, disk, network?)
- What will break first as usage grows?
- Is this a problem for a laptop, a server, or a data center?

### When Simple Architectures Suffice

If your system serves 1,000 requests per day, you do not need Kubernetes, Kafka, or a microservices architecture. A single server with a database is plenty.

Scale estimation protects you from **over-engineering**.

---

## 2. Resource Thinking

### The Four Key Resources

Every system consumes resources. Understanding which one is the bottleneck drives architectural decisions.

| Resource | What It Limits | Typical Bottleneck |
|---|---|---|
| **Storage** | How much data you can keep | Photos, videos, logs, embeddings |
| **Network / Bandwidth** | How much data you can move | Streaming, file transfers, APIs |
| **Compute (CPU)** | How much processing you can do | Inference, ranking, aggregation |
| **Memory** | How much data you can hold in RAM | Caching, model weights, KV cache |

### WhatsApp Resource Example

For a messaging app, all four matter:

- **Storage:** Billions of messages must be persisted
- **Network:** Every message traverses the network (sender → server → recipient)
- **Compute:** Encryption, protocol handling, push notifications
- **Memory:** Active connections, message queues, recent conversation caches

---

## 3. Capacity vs Throughput

### Two Different Concepts

| | Capacity | Throughput |
|---|---|---|
| **Definition** | How much data **exists** | How much work **arrives over time** |
| **Analogy** | Size of a lake | Rate of water flowing into it |
| **Unit** | Bytes, records, documents | Requests/second, MB/second |
| **Question** | "How big is my database?" | "How many queries per second?" |

### Why the Distinction Matters

- A system with **high capacity** but **low throughput** needs big storage but modest compute (e.g., photo archive).
- A system with **low capacity** but **high throughput** needs fast compute but modest storage (e.g., real-time bidding).
- A system with **both high** needs everything at scale (e.g., WhatsApp, YouTube).

### Mental Model: Lake vs Water Flow

- **Lake** = your database (capacity)
- **River feeding the lake** = your incoming requests (throughput)
- A lake can be enormous but fed by a tiny stream.
- A small lake can be fed by a massive waterfall.

Design for whichever is larger or grows faster.

---

## 4. WhatsApp Estimation Example

### Step-by-Step Back-of-the-Envelope

#### Users
- 1 billion active users

#### Messages per User
- 100 messages per day per user (average)

#### Total Messages
```
1B users × 100 messages/day = 100B messages/day
```

#### Message Size
- Average message: ~100 bytes (text only, excluding metadata)

#### Daily Storage
```
100B messages × 100 bytes = 10TB/day
```

#### Annual Storage
```
10TB/day × 365 ≈ 3.6PB/year
```

### What This Tells Us

- **Storage:** 3.6PB/year is data-center scale. A single machine cannot hold this.
- **Throughput:** 100B/day ≈ 1.2M messages/second average
- **Compute:** Encryption and routing for 1.2M msgs/sec is significant

### Purpose of the Math

The goal is **not** to produce exact numbers. The goal is to understand whether a single machine remains practical.

**Conclusion for WhatsApp:** No. Distributed systems are required.

---

## 5. Peak vs Average Traffic

### Humans Are Bursty

Average traffic is misleading. Real-world traffic has spikes:

- **Good morning messages:** 7-9 AM local time spikes
- **Office hours:** Sustained high throughput during work hours
- **Sporting events:** Massive spikes during goals, match ends
- **Holidays:** Family group chats explode

### WhatsApp Example

| Metric | Value |
|---|---|
| Average throughput | ~1.2M messages/second |
| Peak traffic (assumed 10x) | ~12M messages/second |

### Design Implication

Your system must handle **peak load**, not average load. If you design for average:

- Peak traffic causes cascading failures
- Queues back up
- Latency spikes
- Users see timeouts

**Rule of thumb:** Design for 3-10x average, depending on burstiness. Know your domain.

---

## 6. Scale Up vs Scale Out

### Two Strategies for Growth

| | Scale Up (Vertical) | Scale Out (Horizontal) |
|---|---|---|
| **What** | Bigger machine | More machines |
| **Limit** | Hardware ceiling (single box max) | Coordination overhead |
| **Cost** | Expensive hardware | Commodity hardware |
| **Complexity** | Low (still one box) | High (distributed systems) |
| **When to use** | Small scale, temporary | Large scale, permanent |

### Scale Up

Buy a bigger server:
- More CPU cores
- More RAM
- Faster disk (NVMe, SSD)

**Limit:** There is always a biggest machine. Eventually you hit the ceiling.

### Scale Out

Add more servers:
- Shard data across machines
- Load balance requests
- Replicate for availability

**Limit:** Coordination becomes expensive. At some point, managing distributed state dominates.

### When Distributed Systems Emerge

Distributed systems are not a choice — they are a **consequence** of needing more than one machine can provide.

Once a single machine becomes the bottleneck for **any** resource (CPU, memory, disk, network), you must distribute.

### The First Distributed Systems Problem

Once multiple servers exist, you need:

1. **User-to-server mapping (registry):** Which server handles which user?
2. **Server communication:** How do servers talk to each other?
3. **Synchronization:** How do you keep state consistent across servers?
4. **Failure handling:** What happens when a server dies?

These are the foundational challenges of distributed systems.

---

## 7. Key Lessons

### 1. Estimation reveals constraints.

You cannot architect without knowing scale. A system for 1K users looks nothing like a system for 1B users.

### 2. Architecture should emerge from scale.

Do not start with "I will use microservices." Start with "How many users? How much data? What throughput?" and let the architecture follow.

### 3. Peak traffic matters.

Average traffic is for billing. Peak traffic is for architecture. Design for the spikes, not the calm.

### 4. Coordination becomes a challenge after scaling out.

The hard part of distributed systems is not running multiple machines — it is making them agree on state, handle failures, and remain consistent.

---

## Interview Checklist (Scale Estimation)

Before proposing architecture:

- [ ] How many users? (DAU, MAU, total)
- [ ] What is the average throughput? (QPS, messages/sec, requests/sec)
- [ ] What is the peak throughput? (3x? 10x?)
- [ ] How much data is generated per user per day?
- [ ] What is the total storage requirement?
- [ ] What is the dominant resource? (CPU, memory, disk, network?)
- [ ] Can a single machine handle this? If not, what must be distributed?
- [ ] What is the read-to-write ratio?

---

## L5 Additions: ML/GPU Scale Estimation

### GPU Memory Estimation

```
GPU Memory = Model Weights + KV Cache + Activations + Framework Overhead

Model Weights:
  FP16:  2 bytes × N_params
  INT8:  1 byte  × N_params
  INT4:  0.5 bytes × N_params

KV Cache (per token, per layer):
  2 × n_kv_heads × head_dim × 2 bytes (FP16)
  
  LLaMA-2 7B:  2 × 32 × 128 × 2 = 16 KB/token/layer × 32 layers = 512 KB/token
  LLaMA-2 70B: 2 × 8 × 128 × 2 = 4 KB/token/layer × 80 layers = 320 KB/token (GQA)
```

### LLM Serving Capacity Example

```
Model: LLaMA-2 70B (FP16)
GPU: 4x A100 80GB (tensor parallelism = 4)

Weights:  140 GB / 4 GPUs = 35 GB/GPU
Available for KV cache per GPU: 80 - 35 - 5 (overhead) = 40 GB

KV cache per token (with GQA): 320 KB/token
Max concurrent tokens per GPU: 40 GB / 320 KB = 125,000 tokens
Total across 4 GPUs: 500,000 tokens

If average context = 2048 tokens:
  Max concurrent requests = 500,000 / 2048 ≈ 244 requests

If average response = 200 tokens at 50 tokens/sec:
  Throughput = 244 × 50 = 12,200 tokens/sec
```

### Embedding/Vector DB Estimation

```
Documents: 10M documents
Chunk size: 512 tokens → ~20M chunks (avg 2 chunks/doc)
Embedding dim: 1024 (bge-large)
Storage per vector: 1024 × 4 bytes (FP32) = 4 KB

Total vector storage: 20M × 4 KB = 80 GB
HNSW index overhead: ~1.5x → 120 GB
Metadata storage: ~20 GB (Postgres)

Total: ~140 GB — fits on a single machine with 256 GB RAM
```

### Training Compute Estimation

```
Training FLOPs ≈ 6 × N_params × N_tokens

Example: Fine-tuning 7B model on 1B tokens
  FLOPs = 6 × 7B × 1B = 4.2 × 10^19 FLOPs
  
A100 GPU: ~312 TFLOPS (FP16)
  Time = 4.2 × 10^19 / (312 × 10^12) = 134,615 seconds ≈ 37 hours
  
With 8 GPUs (data parallel): ~4.7 hours
With gradient checkpointing (2x compute): ~9.4 hours
```

### Inference Cost Estimation

```
Model: 70B (FP16)
GPU: A100 80GB (cost: ~$2/hour on cloud)

Tokens per second (batch=32): ~2000 tokens/sec
Cost per 1M tokens: 
  $2/hour / (2000 × 3600) = $0.0000278 per token
  = $27.80 per 1M tokens

With INT4 quantization (3x throughput):
  = $9.27 per 1M tokens

With model routing (80% to 8B, 20% to 70B):
  8B cost: ~$2.78 per 1M tokens
  Weighted average: 0.8 × 2.78 + 0.2 × 27.80 = $7.78 per 1M tokens
```

### L5 Interview Q&A

#### Q: "Estimate the GPU requirements for serving a 70B model to 1M daily users."

```
1M users × 5 queries/day = 5M queries/day
Average: 500 prompt tokens + 200 response tokens = 700 tokens/query
Total tokens/day: 5M × 700 = 3.5B tokens/day

Peak QPS (5x average): 5M / 86400 × 5 ≈ 290 QPS

70B model on A100 80GB (TP=4):
  Throughput with continuous batching: ~2000 tokens/sec per 4-GPU node
  Nodes needed for peak: 290 QPS × 700 tokens / 2000 = ~102 nodes

  That's 408 A100 GPUs — very expensive.

Optimization:
  - INT4 quantization: 3x throughput → 34 nodes (136 GPUs)
  - Model routing (80% to 8B): 80% handled by 8B (1 GPU each)
    - 8B nodes: 290 × 0.8 × 700 / 4000 = ~41 nodes (41 GPUs)
    - 70B nodes: 290 × 0.2 × 700 / 6000 = ~7 nodes (28 GPUs)
    - Total: 69 GPUs — 6x reduction
```

#### Q: "How much does it cost to train a 7B model from scratch?"

```
Chinchilla optimal: 7B params × 20 tokens/param = 140B tokens

FLOPs = 6 × 7B × 140B = 5.88 × 10^21 FLOPs

A100 GPU: 312 TFLOPS (FP16, with MFU ~50%): 156 TFLOPS effective
8x A100 node: 1.25 PFLOPS effective

Time = 5.88 × 10^21 / (1.25 × 10^15) = 4.7 × 10^6 seconds ≈ 54 days

Cost: 8 GPUs × $2/hour × 24 × 54 = $20,736

But with LLaMA-style overtraining (2T tokens):
  FLOPs = 6 × 7B × 2T = 8.4 × 10^22
  Time = 8.4 × 10^22 / 1.25 × 10^15 = 67M seconds ≈ 778 days (single node)
  With 64 nodes (512 GPUs): ~12 days
  Cost: 512 × $2 × 24 × 12 = $294,912
```

#### Q: "How do you estimate vector DB storage for 100M documents?"

```
100M documents → ~300M chunks (avg 3 chunks/doc)
Embedding: 1024 dim, FP32 = 4 KB/vector
Vector storage: 300M × 4 KB = 1.2 TB
HNSW index overhead: 1.5x → 1.8 TB
Metadata (Postgres): ~150 GB

Total: ~2 TB

With INT8 quantized embeddings: 500 GB + 750 GB index = 1.25 TB
With INT4 quantized embeddings: 250 GB + 375 GB index = 625 GB

Shard across 4 machines: 156 GB/machine (INT8) — fits in 256 GB RAM
```
