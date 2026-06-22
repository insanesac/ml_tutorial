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
