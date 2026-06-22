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
