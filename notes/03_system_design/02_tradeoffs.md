# System Design Foundations - Session 2

Built through discussion and exercises.

---

## 1. Tradeoffs

### The Core Truth of System Design

System design is rarely about finding the perfect solution. It is about **choosing what to optimize** under constraints.

Fast, accurate, reliable, scalable, and cheap **rarely coexist perfectly**.

> **Key Lesson:** Requirements drive tradeoffs.

### Why Tradeoffs Are Inevitable

Every engineering decision has a cost:
- Lower latency often means less accuracy (caching, approximation)
- Higher availability often means more complexity (replication, consensus)
- Lower cost often means less performance (shared resources, batching)

Understanding which dimension matters most for **this specific system** is what separates junior from senior engineers.

### Real-World Tradeoff Examples

#### Medical Assistant
| Priority | Why |
|---|---|
| **Accuracy dominates latency** | Wrong medical advice can cause serious harm |
| Latency is secondary | A correct slow answer is better than a fast wrong one |

Users would rather wait 5 seconds for a correct diagnosis than get an instant hallucination.

#### Gmail Smart Reply
| Priority | Why |
|---|---|
| **Latency dominates** | Users see suggestions while typing; they must appear instantly |
| Accuracy is secondary | Users can easily ignore or edit bad suggestions |

A slightly imperfect but instant suggestion is better than a perfect one that arrives after the user has already moved on.

#### Coding Assistant (e.g., GitHub Copilot)

This is a nuanced case where the "right" answer depends on context:

- **Low latency** feels magical — suggestions appear as you type
- **Higher accuracy** means less time fixing bad suggestions

**The real objective:** Developer productivity (time to correct working code)

Sometimes a slightly wrong suggestion that is fast to accept and tweak beats a perfect suggestion that arrives too late.

### Interview Framing

When asked to design a system, explicitly state your tradeoffs:

> "For this system, I am prioritizing X over Y because [business reason]. I acknowledge this means [tradeoff consequence]."

This shows structured thinking and business awareness.

---

## 2. Cost of Error

### Not All Mistakes Are Equal

The impact of a wrong answer varies dramatically by domain. Understanding this helps calibrate how much to invest in accuracy, testing, and safety mechanisms.

### Error Cost Spectrum

| System | Cost of Error | Example |
|---|---|---|
| **Autocomplete** | Low | User sees "thnaks" instead of "thanks" — trivial to correct |
| **Chatbot (general)** | Moderate | Wrong restaurant recommendation — annoying but not harmful |
| **Medical Assistant** | Very High | Wrong diagnosis recommendation — potentially life-threatening |
| **Weather Trading Bot** | Direct Monetary | Wrong weather signal → bad trade → real financial loss |
| **Banking Transaction** | Very High | Wrong balance or failed transfer → legal liability |
| **Content Moderation** | High | False negative → harmful content spread; false positive → censorship |

### How Error Cost Drives Design

**Low cost of error** (autocomplete):
- Can use simpler, faster models
- Less need for guardrails and human review
- Aggressive caching acceptable

**High cost of error** (medical, banking):
- Need confidence thresholds
- Human-in-the-loop for edge cases
- Extensive testing and validation
- Audit trails for every decision

### Interview Sound Bite

> "The cost of error in this domain is [high/moderate/low]. This means I would [specific design choice] to mitigate that risk."

---

## 3. Success Metrics

### The Hierarchy of Metrics

A common mistake is optimizing engineering metrics before understanding business goals.

```
Business Goal
      ↓
Product Metric
      ↓
Engineering Metric
```

**Business goals** are what leadership cares about.
**Product metrics** are how PMs measure progress.
**Engineering metrics** are how engineers optimize.

The danger: engineers often optimize the bottom metric without checking if it moves the top one.

### Examples

#### Weather Trading Bot
| Level | Metric |
|---|---|
| Business Goal | **Profit** |
| Product Metric | Signal accuracy, trade execution rate |
| Engineering Metric | Latency, model inference time |

Optimizing for sub-100ms inference is useless if the model makes bad predictions. The business goal is profit, not speed.

#### WhatsApp
| Level | Metric |
|---|---|
| Business Goal | **Adoption / Active Users** |
| Product Metric | Messages sent, retention rate |
| Engineering Metric | Message delivery latency, sync speed |

If messages are fast but users leave because the app is unreliable, the engineering metric was optimized in isolation.

#### Netflix Recommendations
| Level | Metric |
|---|---|
| Business Goal | Revenue / Retention |
| Product Metric | **Watch Time**, completion rate |
| Engineering Metric | Recommendation latency, model freshness |

Watch time is the product metric because it directly predicts retention and subscription value.

#### Enterprise RAG
| Level | Metric |
|---|---|
| Business Goal | **Reduce support load and save employee time** |
| Product Metric | Ticket deflection rate, employee satisfaction |
| Engineering Metric | Retrieval latency, embedding update frequency |

If the RAG system is fast but gives wrong answers, employees still file tickets — the business goal is missed.

#### Coding Assistant
| Level | Metric |
|---|---|
| Business Goal | Developer productivity, feature velocity |
| Product Metric | **Developer Productivity** (time to correct working code) |
| Engineering Metric | Suggestion latency, model accuracy |

The real question is not "how fast are suggestions?" but "do developers ship code faster with this tool?"

### Anti-Pattern: Optimizing the Wrong Metric

> "Our p99 latency improved from 200ms to 50ms!"

Great. But did user retention improve? Did revenue increase? Did support tickets drop?

Always connect engineering metrics upstream to business impact.

---

## 4. Key Mental Models

When approaching any system design problem, ask:

### The Five Questions

1. **What is the business trying to achieve?**
   - Profit? Adoption? Retention? Cost reduction?
   - This is your north star.

2. **What metric proves success?**
   - Not "what metric is easy to measure?"
   - "What metric, if it improves, proves the business is winning?"

3. **What tradeoff am I making?**
   - Every choice gives up something.
   - State it explicitly.

4. **What happens if the system is wrong?**
   - Cost of error drives safety investment.
   - Medical system vs autocomplete have very different answers.

5. **What happens if the system is slow?**
   - Some systems die from latency (trading bots).
   - Some tolerate it (batch analytics).
   - Know which one you are building.

---

## 5. Lessons Learned

### 1. There is rarely a universally correct design choice.

The "best" architecture depends on context. A design that is perfect for WhatsApp might be terrible for a weather trading bot.

### 2. The correct architecture depends on the objective.

Start with the goal, not the technology. The same building blocks (databases, queues, caches) assembled differently serve different purposes.

### 3. Business metrics are usually more important than engineering metrics.

A system with 99th percentile latency but no users is worse than a system with 500ms latency and millions of happy users.

### 4. Strong system designers challenge assumptions and ask what is actually being optimized.

When someone says "we need sub-100ms latency," ask: **why?** What business outcome depends on it? Sometimes the answer reveals the real constraint.

### 5. User productivity, retention, adoption, and profit often matter more than model accuracy.

In ML systems especially, a 95% accurate model that users love beats a 99% accurate model that is too slow or too fragile to use.

---

## Interview Checklist (Tradeoffs & Metrics)

Before proposing any architecture:

- [ ] What is the business goal?
- [ ] What is the product metric that matters?
- [ ] What is the cost of being wrong?
- [ ] What is the cost of being slow?
- [ ] Which NFRs are critical and which are nice-to-have?
- [ ] What tradeoffs am I making, and why?
- [ ] Can I defend this choice if challenged?

---

## L5 Additions: ML-Specific Tradeoffs

### The ML Tradeoff Quadrilateral

```
         Quality (accuracy, faithfulness)
                /\
               /  \
              /    \
             /      \
            /________\
   Latency          Cost
```

In ML systems, there's often a **fourth dimension**: **Freshness** (how current is the model's knowledge).

| Tradeoff | Example |
|---|---|
| Quality vs Latency | Larger model = better answers but slower |
| Quality vs Cost | GPT-4 quality at 100x cost of 8B model |
| Latency vs Cost | GPU acceleration = faster but expensive |
| Quality vs Freshness | Fine-tuned model = high quality but stale knowledge |
| Freshness vs Cost | Real-time RAG indexing = fresh but expensive infra |

### ML System Tradeoff Examples

#### Search Ranking System

| Priority | Why |
|---|---|
| **Latency dominates** | Search results must appear in <100ms |
| Quality is secondary | Users scroll past imperfect results |
| Cost is moderate | Can't spend $0.10 per search |

Architecture: two-stage retrieval (fast candidate generation → slower reranker on top 50).

#### Medical Diagnosis LLM

| Priority | Why |
|---|---|
| **Quality dominates everything** | Wrong diagnosis = patient harm |
| Latency is secondary | Doctors will wait 30 seconds for a correct answer |
| Cost is secondary | Healthcare costs are high anyway |

Architecture: large model + RAG on medical literature + human-in-the-loop review + confidence thresholds.

#### Real-time Fraud Detection

| Priority | Why |
|---|---|
| **Latency dominates** | Transaction must be approved/blocked in <50ms |
| Quality matters but precision > recall | False positives block legitimate purchases (angry customers) |
| Cost is moderate | Fraud losses > compute costs |

Architecture: gradient-boosted trees (fast inference) + feature store for real-time features + fallback to rules-based system.

#### Code Generation Assistant

| Priority | Why |
|---|---|
| **Latency + Quality both critical** | Must be fast AND correct |
| Cost is secondary | Developer productivity >> API costs |
| Freshness matters | New APIs, libraries appear constantly |

Architecture: medium model (13B-70B) + RAG on documentation + fine-tuned on code + speculative decoding for speed.

### Cost of Error in ML Systems

| System | False Positive Cost | False Negative Cost | Optimal Strategy |
|---|---|---|---|
| **Spam filter** | Legitimate email blocked | Spam reaches inbox | Lean toward recall (let some spam through) |
| **Fraud detection** | Legitimate transaction blocked | Fraud goes through | Balance — both are expensive |
| **Medical diagnosis** | Unnecessary treatment | Missed diagnosis | Lean toward recall (better safe than sorry) |
| **Content moderation** | Legitimate content removed | Harmful content stays | Depends on platform values |
| **LLM hallucination** | Over-cautious response | False information given | Lean toward precision (refuse rather than hallucinate) |

### ML-Specific Metrics Hierarchy

```
Business Goal (revenue, retention, cost savings)
      ↓
Product Metric (task completion, user satisfaction, deflection rate)
      ↓
Model Metric (accuracy, faithfulness, hallucination rate, F1)
      ↓
System Metric (latency, throughput, GPU utilization, cost/request)
```

**Anti-pattern**: "Our model accuracy improved from 92% to 95%!"
**Question**: Did user retention improve? Did support tickets decrease? If not, the 3% accuracy improvement was irrelevant.

### L5 Interview Q&A

#### Q: "Design a system where both latency and quality are critical."

**Example: Coding assistant (Copilot-style)**

The key insight is that you need **both** — a slow perfect suggestion is useless (developer has moved on), and a fast wrong suggestion wastes time (developer must debug).

**Solution: Cascading models**
1. **Fast path**: 3B model generates suggestion in <100ms for 90% of cases.
2. **Slow path**: If the 3B model's confidence is low, use 70B model (<500ms).
3. **Speculative decoding**: 3B drafts, 70B verifies — get 70B quality at 3B speed.

**Result**: p50 latency = 100ms (3B), p95 = 500ms (70B fallback), quality ≈ 70B model.

#### Q: "How do you decide between RAG and fine-tuning?"

| Factor | RAG | Fine-tuning |
|---|---|---|
| Knowledge changes frequently | ✅ Real-time updates | ❌ Needs retraining |
| Need citations/sources | ✅ Built-in | ❌ No source tracking |
| Domain-specific style/format | ❌ Prompt engineering only | ✅ Learned behavior |
| Latency critical | ❌ Retrieval adds 50-200ms | ✅ Same inference time |
| Large knowledge base | ✅ Scales with vector DB | ❌ Limited by training data |
| Consistent behavior | ❌ Depends on retrieval quality | ✅ Deterministic |

**Rule of thumb**: RAG for knowledge, fine-tuning for behavior. Use both in production.

#### Q: "Your LLM system has 95% accuracy but users are unhappy. What do you investigate?"

1. **Are you measuring the right metric?** 95% accuracy on what? If it's accuracy on a benchmark but not on real user queries, the metric is wrong.
2. **What's the 5% failure mode?** If the 5% errors are in the most common query types, users will hit them constantly.
3. **Latency**: Is the system too slow? Users may prefer a faster, slightly less accurate system.
4. **Hallucination rate**: Even 5% hallucination can destroy trust — one confident wrong answer is worse than ten "I don't know" responses.
5. **User expectations**: Are users expecting 100% accuracy? Manage expectations through UI (confidence indicators, citations).
6. **Task completion**: Are users actually completing their tasks? Accuracy ≠ task completion.

#### Q: "How do you set confidence thresholds for an LLM system?"

```
High confidence (>0.9):  Auto-respond, no human review
Medium confidence (0.5-0.9):  Respond with "I think..." + provide sources
Low confidence (<0.5):  Refuse or escalate to human

Threshold tuning:
  - Start conservative (high threshold) to build trust
  - Monitor false positive rate (confident but wrong)
  - Gradually lower threshold as model improves
  - Different thresholds for different risk levels:
    - Informational query: threshold = 0.3
    - Action-taking query: threshold = 0.8
    - Safety-critical query: threshold = 0.95
```
