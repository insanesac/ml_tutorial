# System Design Foundations - Session 4

Distributed Systems Foundations Through Thought Experiments

---

## 1. Why Replication Exists

### The Single-Copy Problem

A single copy of data is risky.

**Scenario:**
- One copy of your data lives on Server A
- Server A's hard drive dies
- **Your data is gone**

Machines fail regularly. Hard drives fail. Memory corrupts. Power outages happen. Data centers flood.

### What Replication Solves

Replication means keeping **multiple copies** of important data so the system can survive failures.

```
One copy  -> Machine dies -> Data may be lost
Multiple copies -> Machine dies -> Data survives elsewhere
```

### Replication Is Not Optional at Scale

At small scale, you might get away with backups. At large scale, replication is the only way to ensure:
- **Durability:** Data survives failures
- **Availability:** Users can still access data when some machines are down
- **Read scaling:** Multiple copies can serve read traffic

---

## 2. Durability vs Reaching a Server

### A Critical Distinction

A message **reaching** a server does **not** necessarily mean it is safely stored.

### Possible States After a Message Arrives

| State | Durability | What Happens on Crash |
|---|---|---|
| Message reached server but was never stored | **None** | Message is lost |
| Stored in memory only | **Low** | Message lost on process restart |
| Stored on disk (not fsynced) | **Medium** | Message may be lost on OS crash |
| Stored on disk (fsynced) | **High** | Message survives most crashes |
| Stored and replicated to multiple servers | **Very High** | Message survives single-server failure |

### The Key Question

> **When can the system safely say "Success"?**

The answer depends on your durability requirements:
- **Chat app:** Acknowledge after write to disk + replicate
- **Analytics log:** Acknowledge after memory buffer (batch flush later)
- **Banking transaction:** Acknowledge after fsync + multi-server replication + consensus

### Interview Trap

> "The message was sent successfully."

Ask: **What does "success" mean?** Stored in memory? On disk? Replicated? The answer reveals the system's durability guarantees.

---

## 3. Acknowledgements and Tradeoffs

### Two Acknowledgement Strategies

#### Option A: Acknowledge Immediately

```
Client → Server → "OK" (immediately)
         ↓
    (store asynchronously)
```

- **Pro:** Fast response. Low latency.
- **Con:** Risky. If server crashes before storage, the message is lost but the client thinks it succeeded.

#### Option B: Wait Until Safely Stored

```
Client → Server → Store → Replicate → "OK"
```

- **Pro:** Safe. Message is durable before client is told success.
- **Con:** Slower. Extra disk writes and network round trips.

### Domain-Specific Choice

| System | Typical Strategy | Why |
|---|---|---|
| **WhatsApp** | Wait for replication | Losing messages breaks trust. An extra 200ms is acceptable. |
| **Analytics logging** | Ack immediately | Lost log lines are tolerable. Speed matters more. |
| **Banking** | Wait for consensus | Financial correctness is non-negotiable. |
| **Metrics / telemetry** | Ack immediately | Brief gaps in monitoring are acceptable. |

### WhatsApp Example

For WhatsApp, waiting an extra 200ms is **often preferable to losing messages**.

Users notice missing messages. They rarely notice 200ms latency.

---

## 4. Coordination

### The Coordination Challenge

When data exists on multiple servers, the system must coordinate.

### Key Coordination Questions

| Question | Why It Matters |
|---|---|
| **Did both servers save the data?** | If one failed, the user might get inconsistent results |
| **Which copy is authoritative?** | If copies diverge, which one is "truth"? |
| **When should success be reported to the user?** | Too early = risk. Too late = slow. |

### Coordination Is Expensive

Every coordination step involves:
- **Network round trips** (latency)
- **Consensus logic** (complexity)
- **Failure handling** (what if the coordinator dies?)

**Key principle:** Minimize coordination. Coordinate only when necessary.

---

## 5. The Network Failure Problem

### The Fundamental Uncertainty

In distributed systems, if Server A cannot reach Server B, A cannot know with certainty:

- **Did B crash?**
- **Did the network fail?**
- **Did A lose connectivity?**

From A's perspective, all three look identical: **no response**.

### Why This Matters

If A assumes B is dead and takes over B's responsibilities:
- What if B is actually alive and also serving requests?
- Now you have **two servers doing the same work** on the same data.
- **Data divergence** is inevitable.

If A waits forever for B:
- A blocks indefinitely.
- The system **stops serving requests**.
- **Availability is sacrificed** for safety.

### Distributed Systems Reality

> **Distributed systems often operate with incomplete information.**

This is not a bug. It is a fundamental property of networked systems. All distributed system protocols (consensus, leader election, timeouts) exist to handle this uncertainty.

---

## 6. Timeouts

### Why Timeouts Exist

Because certainty is impossible, systems use timeouts:

```
Wait for some time.
If no response arrives, assume failure and continue.
```

### Critical Distinction

> **Assuming failure is not the same as knowing failure.**

When a timeout fires:
- The remote server **might** be dead
- The remote server **might** be slow
- The network **might** be congested
- Your own network card **might** be broken

You **do not know**. You **assume**.

### Timeout Tradeoffs

| Timeout Duration | Risk |
|---|---|
| Too short | False positives — healthy servers marked as dead, causing unnecessary failover |
| Too long | Slow failure detection — users wait, cascading delays |
| Just right | Detects real failures quickly without excessive false positives |

There is no universally "right" timeout. It depends on network characteristics, load patterns, and acceptable failure detection delay.

---

## 7. Availability vs Consistency

### The Network Partition Scenario

Imagine:
- Server A and Server B cannot communicate (network partition)
- Both have copies of the same data
- A user sends a write to A
- Another user sends a read to B

### Option: Allow Both to Continue

```
Server A accepts writes → becomes divergent
Server B accepts reads → serves stale data
```

- **Benefit:** High availability. Users can still read and write.
- **Risk:** Different versions of reality emerge. Inconsistency.

### The CAP Theorem Connection

When a network partition happens, you must choose:

- **Availability (AP):** Continue serving requests. Risk inconsistency.
- **Consistency (CP):** Reject requests until partition heals. Risk unavailability.

You cannot have both during a partition.

---

## 8. Split Brain

### What Is Split Brain?

Both servers accept changes independently during a partition.

**Example:**
- Server A stores: `profile_name = "Alice"`
- Server B stores: `profile_name = "Bob"`

When communication returns:

> **Which value is correct?**

This is called a **split-brain situation**.

### Why Split Brain Is Dangerous

- **Data loss:** If you pick one value, you lose the other.
- **User confusion:** Different users see different states.
- **Business impact:** Financial records, inventory counts, game scores — all can diverge.

### Prevention Strategies

| Strategy | How It Works | Tradeoff |
|---|---|---|
| **Leader-based systems** | Only the leader accepts writes. Followers are read-only. | Leader becomes a bottleneck and single point of failure |
| **Quorum-based writes** | Write must be acknowledged by majority. | Higher latency, more coordination |
| **Conflict resolution** | Store both versions, resolve later (CRDTs, vector clocks). | Complex, eventual consistency |

---

## 9. Different Systems, Different Tradeoffs

### Domain Determines the Right Choice

| Domain | Preferred Approach | Why |
|---|---|---|
| **Messaging (WhatsApp)** | Temporary inconsistency may be acceptable | Users prefer receiving a message eventually over the system being down |
| **Banking** | Temporary inconsistency can be catastrophic | Incorrect balances are legally and financially unacceptable |
| **Social media likes** | Eventual consistency is fine | A like count that is slightly off is not harmful |
| **E-commerce inventory** | Strong consistency preferred | Selling the same item twice = unhappy customers |

### Key Lesson

> The right tradeoff depends on the **business domain**, not on abstract principles.

What is correct for a chat app is wrong for a bank. Context matters.

---

## 10. Leaders

### The Chain of Command Analogy

Organizations often have chains of command:
- One person makes the final call
- Others follow or advise

Distributed systems use the same idea.

### What Is a Leader?

A **leader** is a single authority for important decisions:
- Accepts all writes
- Propagates changes to followers
- Resolves conflicts

### Benefits of Leader-Based Design

| Benefit | Explanation |
|---|---|
| **Point of truth** | One authoritative copy prevents divergence |
| **Reduced conflicts** | No split brain — single writer |
| **Clear decision making** | Simple mental model for coordination |

### Common Leader-Based Systems

- **Primary-replica databases:** Primary handles writes, replicas handle reads
- **Raft / Paxos consensus:** Leader coordinates log replication
- **Kafka:** One broker is the leader for each partition

---

## 11. Leader Failure

### What Happens When the Leader Crashes?

| Effect | Consequence |
|---|---|
| New writes may pause | No leader to accept them |
| Important decisions cannot be made | Consensus stalls |
| Reads may still continue | Replicas can serve stale reads |

### The New Question

> **Who becomes the next leader?**

This is the **leader election problem**.

### Leader Election Is Hard

- Multiple followers might think they should be leader
- The old leader might not actually be dead (network partition)
- Electing the wrong leader causes split brain

### Solutions

| Approach | Examples |
|---|---|
| **External coordinator** | ZooKeeper, etcd (consensus-based election) |
| **Raft / Paxos** | Built-in leader election with safety guarantees |
| **Manual failover** | Human decides (slow, error-prone) |

---

## 12. Key Mental Models

### Think User First

> **What does the user observe?**

Users do not care about your Raft quorum. They care whether their message sent, their payment went through, their video loaded.

Design from the user experience inward, not from the technology outward.

### Think Business First

> **What is the cost of being wrong?**

A wrong movie recommendation costs nothing.
A wrong medical diagnosis costs lives.
A wrong bank transfer costs money and trust.

The architecture must match the business consequence.

### Distributed Systems Reality

> - Machines fail.
> - Networks fail.
> - Perfect information does not exist.
> - Most solutions are tradeoffs rather than absolutes.

Accepting these realities is the first step to designing robust systems.

---

## Interview Checklist (Distributed Systems)

When discussing any distributed design:

- [ ] What happens when a single server dies?
- [ ] How is data replicated? How many copies?
- [ ] When does the system acknowledge success? (memory? disk? replicated?)
- [ ] What happens during a network partition?
- [ ] Is the system AP or CP? Why?
- [ ] How is leader failure handled?
- [ ] What is the cost of inconsistency in this domain?
- [ ] What is the timeout strategy?
- [ ] Can I explain this in terms of user impact?

---

## L5 Additions: ML-Specific Distributed Systems

### Distributed Training Consistency

#### Data Parallelism

```
GPU 0: Batch 0 → Forward → Backward → Gradients
GPU 1: Batch 1 → Forward → Backward → Gradients
GPU 2: Batch 2 → Forward → Backward → Gradients
GPU 3: Batch 3 → Forward → Backward → Gradients
                    ↓ All-Reduce ↓
              Averaged Gradients
                    ↓
              Update Weights (all GPUs in sync)
```

**Consistency model**: Synchronous — all GPUs must finish before weights update. A slow GPU blocks everyone.

**Failure handling**: If one GPU dies mid-step, the entire step is wasted. Checkpointing every N steps is essential.

#### Model Parallelism (Tensor Parallel)

```
Weight matrix W (D × D) split across GPUs:
  GPU 0: W[:, 0:D/2]
  GPU 1: W[:, D/2:D]

Forward: x @ W = x @ W_0 || x @ W_1 → all-reduce to combine
```

**Consistency**: Weights are sharded — no single GPU has the full model. Communication is required for every forward/backward pass.

#### Pipeline Parallelism

```
GPU 0: Layer 0-19  →  GPU 1: Layer 20-39  →  GPU 2: Layer 40-59  →  GPU 3: Layer 60-79

Micro-batch 1: [GPU0] → [GPU1] → [GPU2] → [GPU3]
Micro-batch 2:       [GPU0] → [GPU1] → [GPU2] → [GPU3]
```

**Pipeline bubble**: GPU 1 waits for GPU 0, GPU 2 waits for GPU 1, etc. The bubble wastes ~50% of compute with 4 GPUs and 1 micro-batch. Mitigated with many micro-batches.

### Model Replication for Inference

```
GPU Group A (4 GPUs): 70B model replica 1
GPU Group B (4 GPUs): 70B model replica 2
GPU Group C (4 GPUs): 70B model replica 3

Load balancer distributes requests across replicas.
Each replica is independent — no coordination needed.
```

**Key difference from training**: Inference replicas are **stateless** (except KV cache). No consensus needed. If a replica dies, the load balancer routes elsewhere.

### KV Cache Consistency

```
Problem: In continuous batching, each request has its own KV cache.
  - No consistency issue (per-request state).
  
Problem: Prefix caching shares KV cache across requests.
  - If the shared prefix changes (e.g., system prompt update), all cached prefixes are invalidated.
  - Solution: version the system prompt; old cache entries expire.
```

### Model Versioning and Rollouts

```
Blue-Green Deployment for Models:
  Blue (current):  v1.0 serving 100% traffic
  Green (new):     v1.1 serving 0% traffic

  1. Deploy v1.1 on separate GPUs (green)
  2. Route 5% traffic to green (canary)
  3. Monitor quality metrics (faithfulness, latency, errors)
  4. If good: ramp to 50% → 100%
  5. If bad: rollback to blue (still running)
```

**Shadow deployment**: Run v1.1 in parallel (not serving users), compare outputs with v1.0. No user impact, but 2x GPU cost.

### Distributed Training Failure Modes

| Failure | Impact | Mitigation |
|---|---|---|
| **GPU dies** | Entire step lost | Checkpoint every N steps, resume from last checkpoint |
| **Network partition** | Some GPUs can't communicate | Timeout + retry, or skip step |
| **Slow GPU (straggler)** | Blocks all GPUs (sync training) | Gradient compression, async training (stale gradients) |
| **OOM on one GPU** | Training crashes | Gradient checkpointing, reduce batch size, FSDP sharding |
| **Checkpoint corruption** | Hours of progress lost | Redundant checkpoints, checksums |

### L5 Interview Q&A

#### Q: "How do you handle a GPU failure during distributed training?"

1. **Checkpointing**: Save model state every N steps (e.g., every 100 steps or 1 hour).
2. **Resume**: Detect failure → load last checkpoint → restart training from there.
3. **Redundant checkpoints**: Store checkpoints on multiple storage systems (NFS + S3).
4. **Fast resume**: Keep optimizer state in checkpoint to avoid re-warmup.
5. **Elastic training**: Some frameworks (TorchElastic) can restart failed workers and rejoin the training group.

**Cost of failure**: If checkpoint is every 1 hour and GPU dies at 59 minutes, you lose 1 hour of compute. With 512 GPUs at $2/hour, that's $1,024 per failure.

#### Q: "Synchronous vs asynchronous distributed training — when to use which?"

| | Synchronous | Asynchronous |
|---|---|---|
| **Consistency** | All GPUs use same gradients | GPUs use stale gradients from peers |
| **Convergence** | Stable, well-understood | Can be unstable (stale gradients) |
| **Throughput** | Limited by slowest GPU | No waiting — max throughput |
| **When to use** | Most training (preferred) | Very large clusters where stragglers are common |

**Modern practice**: Synchronous with gradient compression (reduce communication) and overlap of compute/communication. Async is rarely used for LLM training because stale gradients cause quality issues.

#### Q: "How do you roll out a new model version without downtime?"

1. **Blue-green**: Deploy new version on separate GPUs. Switch traffic when ready.
2. **Canary**: Route small % of traffic to new model. Monitor metrics. Ramp up.
3. **Shadow**: Run new model in parallel, compare outputs (no user impact).
4. **Rolling update**: Replace GPU groups one at a time (some downtime risk).
5. **Multi-LoRA**: If only adapter changed, hot-swap LoRA weights (no GPU restart needed).

**Monitoring during rollout**: faithfulness, hallucination rate, latency, user feedback. Auto-rollback if metrics degrade beyond threshold.

#### Q: "What's the difference between model replication for inference and data parallelism for training?"

**Training (data parallel)**: All replicas compute gradients → all-reduce to sync → update weights. Requires communication and consensus. Replicas must be in sync.

**Inference (model replication)**: Each replica is independent. No communication needed. Load balancer routes requests. If one replica dies, others continue. Stateless (except per-request KV cache).

The key insight: **training requires consensus, inference does not**. This is why inference scaling is much simpler than training scaling.
