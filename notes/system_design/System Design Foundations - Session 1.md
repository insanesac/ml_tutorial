# System Design Foundations - Session 1

Built through discussion and exercises.

---

## 1. Functional Requirements

### What Are Functional Requirements?

Functional requirements are the **capabilities a system must provide** to solve its intended problem. Without them, the solution is incomplete or fails to fulfill its primary purpose.

Think of it this way: if you remove a functional requirement, does the system still solve the user's core problem? If the answer is **no**, then that requirement is **essential**.

### Why They Matter in Interviews

In a system design interview, the interviewer rarely hands you a requirements document. They say something like:

> "Design a URL shortener like bit.ly."

Your first job is to **discover the functional requirements**. If you skip this step and jump straight to hashing algorithms, you miss half the interview.

### Discovery Technique: The Removal Test

The fastest way to identify core vs secondary requirements:

> **If removing a capability destroys the product's purpose, it is likely core.**  
> **If removing it mainly hurts convenience, it is likely secondary.**

This is not about importance in absolute terms — it is about whether the product **still functions** without it.

### Detailed Examples

#### Gmail
| Requirement | Core? | Why? |
|---|---|---|
| Send Email | **Yes** | Without this, it is not an email service |
| Receive Email | **Yes** | Two-way communication is essential |
| Store Email | **Yes** | Users expect persistence |
| Search Email | **Yes** | At scale, browsing is unusable |
| Spam Filtering | No | Important, but Gmail without spam filtering is still Gmail |
| Themes / Customization | No | Purely cosmetic |

#### WhatsApp
| Requirement | Core? | Why? |
|---|---|---|
| Send Message | **Yes** | Core purpose |
| Receive Message | **Yes** | Core purpose |
| Store Message | **Yes** | Users expect chat history |
| Voice Notes | No | Very useful, but not essential to messaging |
| Status Stories | No | Engagement feature, not core |

#### YouTube
| Requirement | Core? | Why? |
|---|---|---|
| Upload Video | **Yes** | Without upload, there is no content |
| Search Video | **Yes** | Discovery is critical at scale |
| Watch Video | **Yes** | Consumption is the point |
| Like / Subscribe | No | Engagement features, not core to watching |
| Comments | No | Community feature, video works without it |

#### Spotify
| Requirement | Core? | Why? |
|---|---|---|
| Upload Music | **Yes** | Content must come from somewhere |
| Search Music | **Yes** | Discovery at scale |
| Listen to Music | **Yes** | Core purpose |
| Playlists | No | Important for UX, but not fundamental |
| Social Sharing | No | Nice to have |

#### Amazon
| Requirement | Core? | Why? |
|---|---|---|
| List Products | **Yes** | Sellers need to display inventory |
| Search Products | **Yes** | Discovery at scale |
| View Products | **Yes** | Users need to see what they are buying |
| Purchase Products | **Yes** | Core transaction flow |
| Reviews | No | Trust signal, but Amazon without reviews still sells |
| Recommendations | No | Conversion driver, not essential |

#### ChatGPT
| Requirement | Core? | Why? |
|---|---|---|
| Ask Questions | **Yes** | Input mechanism |
| Get Responses | **Yes** | Output mechanism — this is the product |
| Conversation History | No | Useful, but single-turn answers still work |
| File Upload | No | Extended feature |

#### Netflix
| Requirement | Core? | Why? |
|---|---|---|
| Search Shows | **Yes** | Discovery at scale |
| Watch Shows | **Yes** | Core purpose |
| Download for Offline | No | Important for mobile, but not core |
| Multiple Profiles | No | Family feature |

### Secondary Requirements Are Still Important

Do not dismiss secondary requirements. In a real interview, after identifying core requirements, you should acknowledge secondary ones and note that they impact **scope and prioritization** but not the fundamental design.

### Interview Trap

> "I would use Redis for caching, Kafka for events, and PostgreSQL for the main store."

This is a **trap** because it jumps to implementation before requirements are clear. Always start with:

1. Who is the user?
2. What do they need to do?
3. What happens if they cannot do it?

---

## 2. Non-Functional Requirements (NFRs)

### What Are NFRs?

NFRs describe **how well** the system performs its job. They are the quality attributes of the system.

If functional requirements define **what** the system does, NFRs define **how well** it does it.

### The Five Core NFRs

#### Latency
**How fast does the system respond?**

- Measured in milliseconds for APIs, seconds for batch jobs.
- **Low latency** means sub-100ms for user-facing requests.
- **High latency** might be acceptable for background processing.
- In ML serving: first-token latency vs time-to-last-token matter differently for different use cases.

**Interview question to ask:**
> "What is the acceptable p99 latency for this endpoint?"

#### Availability
**Is the system up and usable when users need it?**

- Measured as uptime percentage (e.g., 99.9% = ~8.7 hours downtime per year).
- **High availability** systems (99.99%+) require redundancy, failover, and careful design.
- Not every system needs 99.999% availability — it is expensive.

**Interview question to ask:**
> "What is the business cost of 1 hour of downtime?"

#### Reliability
**Is the system correct and trustworthy?**

- Reliability means the system does what it promises, consistently.
- A system can be **available** but **unreliable** (e.g., returning wrong data quickly).
- In ML: reliable means the model outputs are accurate and consistent.

**Interview question to ask:**
> "What happens if the system returns an incorrect result?"

#### Scalability
**Does the system still work when usage grows?**

- **Vertical scaling:** bigger machine (limited).
- **Horizontal scaling:** more machines (the typical cloud approach).
- Scalability is not just about handling more users — it is about handling more **data**, more **requests**, more **geographies**.

**Interview question to ask:**
> "What is the expected growth in users / data / requests over the next year?"

#### Security
**Can users trust the system with their data?**

- Authentication (who are you?), authorization (what can you do?), encryption, audit logging.
- In ML: model inversion attacks, prompt injection, data poisoning.
- Security is often a regulatory requirement (GDPR, SOC2, etc.).

**Interview question to ask:**
> "What data is sensitive, and who should access it?"

### Mental Model for Prioritizing NFRs

Remove the property and ask:

> **What happens to the user?**

This reveals how important the NFR is for **this specific system**.

### Detailed Examples

#### Gmail
| NFR | Priority | Reasoning |
|---|---|---|
| Latency | Medium | Important, but email is not real-time. Users tolerate 1-2 second send delays |
| Availability | High | Email is critical infrastructure for many businesses |
| Reliability | High | Lost emails are unacceptable |
| Scalability | High | Billions of emails per day |
| Security | Critical | Email contains sensitive information |

#### WhatsApp
| NFR | Priority | Reasoning |
|---|---|---|
| Latency | Critical | Messages must feel instant (<200ms ideally) |
| Availability | Critical | People rely on it for real-time communication |
| Reliability | Critical | Lost or out-of-order messages break trust |
| Scalability | Critical | Billions of users, millions of messages per second |
| Security | Critical | End-to-end encryption is a core promise |

#### ATM
| NFR | Priority | Reasoning |
|---|---|---|
| Latency | Medium | Users expect quick response, but seconds are acceptable |
| Availability | Critical | People need cash immediately — failed ATM = angry customer |
| Reliability | Critical | Wrong balance or failed transaction = legal liability |
| Scalability | Low | Per-ATM load is bounded |
| Security | Critical | Physical + digital security both matter |

#### Banking
| NFR | Priority | Reasoning |
|---|---|---|
| Latency | Medium | Transfers can take minutes in many systems |
| Availability | High | But brief maintenance windows are often acceptable |
| Reliability | Critical | Incorrect transactions lose money and trust |
| Scalability | High | Millions of transactions per day |
| Security | Critical | Regulated industry, fraud prevention is essential |

#### Netflix
| NFR | Priority | Reasoning |
|---|---|---|
| Latency | High | Buffering is frustrating, but brief pauses are tolerated |
| Availability | Medium | Short outages are annoying but not catastrophic |
| Reliability | Medium | Wrong recommendation is not a system failure |
| Scalability | Critical | Global scale, peak evening traffic |
| Security | Medium | Account sharing is a bigger issue than data breaches for Netflix |

#### Weather Trading Bot
| NFR | Priority | Reasoning |
|---|---|---|
| Latency | Critical | Trading opportunities disappear in milliseconds |
| Availability | High | Downtime = missed trades |
| Reliability | Critical | Wrong signal = massive financial loss |
| Scalability | Medium | Bounded by market data volume |
| Security | High | API keys, trading capital at risk |

### NFR Tradeoffs

NFRs often conflict:

- **Availability vs Consistency:** CAP theorem territory. Distributed systems must choose.
- **Latency vs Reliability:** More checks = slower but more correct.
- **Security vs Latency:** Encryption and auth add overhead.

In an interview, explicitly discuss tradeoffs. It shows mature system thinking.

---

## Key Lessons Learned

### 1. Start with the user problem, not the technology.

The worst system design answers start with:
> "I would use Kubernetes and Kafka and Redis..."

The best answers start with:
> "Let me understand who the user is and what they need to do."

### 2. Separate WHAT from HOW WELL.

- **Functional requirements** = what the system does
- **Non-functional requirements** = how well it does it

Mixing them leads to vague designs. Be explicit.

### 3. Avoid jumping to implementation details while gathering requirements.

Resist the urge to name databases, queues, or cloud services in the first 5 minutes. Requirements first. Architecture second. Technology third.

### 4. Use the Removal Test to identify core requirements.

For every capability you list, ask:
> "If I remove this, does the product still serve its core purpose?"

This quickly separates essentials from nice-to-haves.

### 5. Evaluate NFRs by considering the business impact when they are lost.

NFR prioritization is **context-dependent**. High availability is critical for an ATM but less critical for a weather blog. Always ground NFRs in user impact and business value.

---

## Interview Checklist (Before You Draw a Single Box)

Before touching the whiteboard or mentioning any technology, confirm:

- [ ] Who is the primary user?
- [ ] What are the core functional requirements? (Removal Test)
- [ ] What are the secondary functional requirements?
- [ ] Which NFRs matter most for this system? (Latency? Availability? Reliability?)
- [ ] What are the rough scale estimates? (Users, QPS, data volume)
- [ ] Are there any regulatory or security constraints?

Only after this checklist should you begin drawing the architecture.

---

## L5 Additions: ML-Specific Requirements

### ML System Functional Requirements

For ML/LLM systems, functional requirements extend beyond traditional software:

| Category | Questions to Ask |
|---|---|
| **Input modality** | Text only? Images? Audio? Multi-modal? |
| **Output format** | Free text? Structured JSON? Code? Citations? |
| **Knowledge scope** | General knowledge? Domain-specific? Real-time data? |
| **Interaction mode** | Single-turn? Multi-turn conversation? Streaming? |
| **Personalization** | Per-user customization? Memory across sessions? |
| **Tool use** | Does the model need to call external APIs/tools? |
| **Safety constraints** | What content must be blocked? What must be flagged? |

### LLM System Examples

#### Enterprise Document Q&A (RAG)

| Requirement | Core? | Why? |
|---|---|---|
| Answer questions from documents | **Yes** | Core purpose |
| Cite sources | **Yes** | Trust — without citations, answers are unverifiable |
| Handle multi-turn follow-ups | **Yes** | Users need conversational refinement |
| Support multiple file formats (PDF, DOCX, HTML) | No | Important but extensible |
| Multi-language support | No | Phase 2 feature |
| Access control per document | No | Important for enterprise but not core to Q&A |

#### Coding Assistant (Copilot-style)

| Requirement | Core? | Why? |
|---|---|---|
| Suggest code completions | **Yes** | Core purpose |
| Context-aware (open files, imports) | **Yes** | Without context, suggestions are useless |
| Low latency (<200ms) | **Yes** | Must appear as you type |
| Explain suggestions | No | Useful but not core to completion |
| Support multiple languages | No | Start with one, expand later |
| Debug assistance | No | Separate feature |

#### Content Moderation System

| Requirement | Core? | Why? |
|---|---|---|
| Classify content as safe/unsafe | **Yes** | Core purpose |
| Handle text + images | **Yes** | Content is multi-modal |
| Provide reason for flagging | **Yes** | Required for appeals process |
| Real-time scoring | No | Batch processing acceptable for many use cases |
| Customizable thresholds | No | Important but configurable post-launch |

### ML-Specific NFRs

| NFR | ML-Specific Concern |
|---|---|
| **Latency** | TTFT vs TPOT for LLMs; model inference time vs retrieval time |
| **Availability** | Model service downtime vs fallback to cached/simpler model |
| **Reliability** | Model correctness — hallucination rate, factual accuracy |
| **Scalability** | GPU memory as bottleneck; batch size vs concurrency |
| **Security** | Prompt injection, data poisoning, model inversion attacks |
| **Freshness** | How often is the model/knowledge base updated? |
| **Cost** | Cost per inference; GPU-hours; API costs for external models |

### L5 Interview Q&A

#### Q: "What questions would you ask to clarify requirements for an LLM-powered customer support chatbot?"

**Functional:**
- What channels (web, mobile, email, API)?
- What types of queries (FAQ, troubleshooting, account management)?
- Does it need to escalate to human agents? When?
- Does it need access to user account data?
- Multi-turn conversation or single-turn Q&A?
- What languages?

**Non-functional:**
- How many concurrent users? Peak QPS?
- What's the acceptable response latency? (TTFT, full response time)
- What's the hallucination tolerance? (Safety-critical vs informational)
- What's the cost budget per query?
- What uptime is required? (24/7 or business hours?)
- What data privacy regulations apply? (GDPR, HIPAA, SOC2?)

**ML-specific:**
- Does it need to cite sources (RAG)?
- Should it learn from user feedback over time?
- What's the process for updating the knowledge base?
- How do we handle queries the model can't answer?

#### Q: "How do you determine if a feature is core vs secondary in an ML system?"

Apply the **Removal Test** with an ML twist:

1. **Without this feature, does the model still produce useful output?** If no → core.
2. **Without this feature, is the output trustworthy?** If no → core (safety/grounding).
3. **Without this feature, is the output fast enough to be usable?** If no → core (latency).
4. **Without this feature, is the system cost-effective?** If no → core (cost constraint).

Example: For a medical advice LLM:
- Grounding in medical literature → **core** (without it, hallucinations could harm)
- Citation of sources → **core** (trust requirement)
- Multi-language support → secondary
- Conversational memory → secondary

#### Q: "What's the biggest mistake candidates make in ML system design interviews?"

**Jumping to model architecture before understanding the problem.**

Bad answer: "I'd use a 70B parameter LLM with RAG and FAISS for retrieval."

Good answer: "Let me first understand who the users are, what they need, and what constraints we have. Then I'll determine whether we need RAG, what model size is appropriate, and how to serve it."

The interviewer wants to see your **thinking process**, not your knowledge of tools. Requirements → Architecture → Technology. Always in that order.
