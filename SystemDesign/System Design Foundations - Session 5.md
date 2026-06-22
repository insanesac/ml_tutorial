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
