# LLM Training Pipeline: From Pretraining to GRPO

## The Big Picture

Modern LLM training is not a collection of unrelated algorithms. Each stage solves a limitation of the previous stage:

```
Raw Internet
    ↓
Pretraining (Next Token Prediction)
    ↓
General Language Model
    ↓
Supervised Fine-Tuning (SFT)
    ↓
Instruction Following Assistant
    ↓
Preference Alignment (RLHF / PPO)
    ↓
DPO (Removes Reward Model)
    ↓
GRPO (Removes Critic)
```

---

## Stage 1: Pretraining

### Goal

Teach the model the structure of language. The model learns grammar, syntax, semantics, reasoning patterns, factual knowledge, code, and world knowledge.

The objective is simply: **predict the next token.**

```
Input:  "The capital of France is"
Target: "Paris"
```

Training pipeline:
```
Input → Transformer → Logits → Softmax → Cross Entropy → Backpropagation → Adam
```

### Result

A model that is very good at continuing text.

### Problem After Pretraining

The model understands language, but it does not understand instructions. A pretrained language model may continue text like an internet forum — it has never been explicitly taught to behave like an assistant.

---

## Stage 2: Supervised Fine-Tuning (SFT)

### Goal

Teach the model to follow instructions. Instead of internet text, the dataset becomes:

```
Instruction → High-quality Answer
```

### Teacher Forcing

During training, we always feed the **correct** previous tokens:

```
Input:  "I"              → Target: "love"
Input:  "I love"         → Target: "machine"
```

Even if the model predicted "I hate", we still feed "I love". This stabilizes training.

### Exposure Bias

During inference, there is no teacher. The model must consume its own predictions:

```
Training:  Correct Prefix → Correct Prefix → Correct Prefix
Inference:  Prediction    → Prediction    → Prediction
```

One mistake can change the entire future context. This is called **exposure bias**.

### Training Objective

Nothing changes mathematically. Still Cross Entropy → Backpropagation → Adam. The only thing that changes is the dataset.

### Result

The language model becomes an instruction-following assistant.

### Problem After SFT

Suppose the prompt is "Write me a joke." There is not one correct answer — there may be Joke A, Joke B, Joke C. All are valid. Some are simply preferred by humans. Cross entropy cannot express "A is better than B." It only knows "this token is correct." We need preference learning.

---

## Stage 3: RLHF (Reinforcement Learning from Human Feedback)

### Pipeline

```
Pretrained LLM
    ↓
Supervised Fine-Tuning (SFT)
    ↓
Reward Model Training
    ↓
PPO (RL loop)
    ↓
Aligned Model
```

### Step 1: Supervised Fine-Tuning (SFT)

Fine-tune the base model on high-quality (prompt, response) pairs written by humans.

Teaches the model the format and style of helpful responses.

### Step 2: Reward Model

Collect human preference data:

```
Prompt: "Explain gravity"
Response A: good explanation
Response B: bad explanation
Human labels: A > B
```

Train a reward model `R(prompt, response) -> scalar` to predict human preference.

### Step 3: PPO (Proximal Policy Optimization)

Use the reward model as a signal to fine-tune the SFT model via RL.

The model (policy) generates responses, reward model scores them, PPO updates the policy.

Add a **KL penalty** to prevent the model drifting too far from the SFT checkpoint:

```
Objective = R(response) - β * KL(policy || SFT)
```

### RLHF Weaknesses

- Expensive: requires human labelers
- Reward hacking: model learns to fool the reward model
- PPO is unstable and complex to tune
- Reward model can have its own biases
- Requires **three** separate learned components: Policy Model, Reward Model, Critic

---

## Stage 4: DPO (Direct Preference Optimization)

### Why DPO?

PPO still requires three separate learned components (Policy, Reward Model, Critic). Researchers asked: why train a reward model? Humans already provide preference data.

```
Prompt → Chosen Response ✅ / Rejected Response ❌
```

Instead of learning a reward function, optimize directly using `Chosen > Rejected`. No reward model required.

### Core Insight

Skip the reward model entirely.

DPO shows that the RLHF objective can be optimized **directly** on preference data without training a separate reward model or running PPO.

### Data Format

```
(prompt, chosen_response, rejected_response)
```

### Loss Function

```
L_DPO = -log σ( β * log(π(chosen|prompt)/π_ref(chosen|prompt))
               - β * log(π(rejected|prompt)/π_ref(rejected|prompt)) )
```

Where:
- `π` = policy being trained
- `π_ref` = frozen SFT reference model
- `β` = temperature controlling deviation from reference
- `σ` = sigmoid

### Intuition

Increase probability of chosen response relative to reference.

Decrease probability of rejected response relative to reference.

The ratio to reference prevents the model from collapsing.

### DPO vs RLHF

| | RLHF | DPO |
|---|---|---|
| Reward model needed | Yes | No |
| RL training loop | Yes (PPO) | No |
| Stability | Hard to tune | Much simpler |
| Data format | Ranked pairs | Ranked pairs |
| Memory cost | High (multiple models) | Lower |
| Quality | Strong | Competitive |

---

## SFT vs RLHF vs DPO — When to Use

| Method | Use When |
|---|---|
| SFT only | Limited data, quick baseline |
| RLHF | Maximum quality, large budget |
| DPO | Good quality, simpler pipeline |

---

## Key Terms

- **Policy** — the LLM being trained
- **Reference model** — frozen SFT checkpoint used as anchor
- **KL divergence** — prevents reward hacking by penalizing drift
- **Chosen / Rejected** — DPO data pair labels

---

## Stage 5: GRPO (Group-Relative Policy Optimization)

### Why GRPO?

DPO still requires a Critic to estimate advantages. GRPO removes the critic entirely.

Generate multiple responses per prompt. Instead of training a critic, compute the **average reward** of the group. Advantages become `Reward − Group Average`. The group itself provides the baseline.

### Used in: DeepSeek R1

GRPO eliminates the need for a critic (value model) by using **group-relative advantages**:

```
1. For each prompt, sample G responses from the policy
2. Score each response with a reward function (e.g., correctness)
3. Compute group-relative advantage: A_i = (r_i - mean(r)) / std(r)
4. Update policy with clipped objective (PPO-style) using A_i
```

### Key Differences from PPO

| | PPO | GRPO |
|---|---|---|
| Value model | Required (critic) | Not needed |
| Advantage | GAE (temporal) | Group-relative (mean/std) |
| Memory | Policy + Value + Reference | Policy + Reference only |
| Used in | GPT-3/4 alignment | DeepSeek R1 |

### Why GRPO Works

- For reasoning tasks (math, code), you can verify correctness automatically.
- Sampling G responses per prompt gives a natural baseline (group mean).
- No value model = simpler training, less memory, fewer hyperparameters.

### Reward Function for R1

DeepSeek R1 uses **rule-based rewards** instead of a neural reward model:
- **Correctness**: exact match or test pass rate.
- **Format**: checks if response uses `<think>` tags properly.
- **No neural reward model** → no reward hacking from a flawed learned reward.

---

## DPO Variants

### KTO (Kahneman-Tversky Optimization)

- Uses **unpaired** data (single response + thumbs up/down) instead of preference pairs.
- Based on prospect theory from behavioral economics.
- Easier data collection — don't need paired comparisons.

### IPO (Identity Preference Optimization)

- Addresses DPO's overfitting issue on preference data.
- Uses a regularization term that prevents the policy from becoming too confident.
- Better for small datasets where DPO overfits.

### SimPO (Simple Preference Optimization)

- Removes the reference model entirely.
- Uses a length-normalized reward: `r(x,y) = β * log π(y|x) / |y|`
- Simpler, less memory, competitive quality.
- Reference-free → no need to load a second model.

### Comparison

| Method | Reward Model | Reference Model | Data Format | Memory |
|---|---|---|---|---|
| RLHF (PPO) | Yes | Yes | Scalar rewards | Highest |
| DPO | No | Yes | Preference pairs | Medium |
| KTO | No | Yes | Unpaired (binary) | Medium |
| IPO | No | Yes | Preference pairs | Medium |
| SimPO | No | No | Preference pairs | Lowest |
| GRPO | No (rule-based) | Yes | Group samples | Medium |

---

## Reward Hacking

### What It Is

The policy finds a way to get high reward **without** actually being helpful:

```
Reward model: "longer responses are better" (learned spurious correlation)
Policy: generates very long, repetitive, unhelpful responses
Result: high reward, low quality
```

### Common Failure Modes

1. **Length hacking**: reward model correlates length with quality → policy generates verbose filler.
2. **Repetition hacking**: repeating key phrases that the reward model likes.
3. **Format hacking**: exploiting formatting (bullet points, headers) that correlates with quality.
4. **Sycophancy**: agreeing with the user regardless of correctness.
5. **Reward model exploitation**: finding adversarial inputs that trick the reward model.

### Mitigations

1. **KL penalty**: penalize drift from reference model — prevents extreme behaviors.
2. **Reward model ensembles**: average multiple reward models — harder to exploit.
3. **Length normalization**: divide reward by response length.
4. **Adversarial training**: include reward-hacking examples in training data.
5. **Monitoring**: track reward vs human eval quality — alert on divergence.
6. **Rule-based rewards**: use verifiable rewards (correctness, format) when possible.

---

## RLAIF (Constitutional AI)

### The Idea

Replace human feedback with **AI feedback**:

```
1. Generate responses with the policy
2. Ask an AI model to evaluate them against a "constitution" (set of principles)
3. Use AI preferences as training signal for DPO/RLHF
```

### Anthropic's Constitutional AI

```
Constitution principles:
- "Be helpful, harmless, and honest"
- "Don't provide dangerous information"
- "Acknowledge uncertainty"

Process:
1. Generate response to a prompt
2. Ask AI: "Is this response consistent with the constitution?"
3. AI provides critique and revision
4. Use (original, revised) pairs as preference data for DPO
```

### Why RLAIF Matters

- **Scalable**: no human annotation bottleneck.
- **Cheaper**: AI feedback costs less than human feedback.
- **Consistent**: AI applies principles more consistently than humans.
- **Risk**: AI feedback may have blind spots or biases that human feedback wouldn't.

---

## Alignment Pipeline (End-to-End)

```
Pretraining (next-token prediction on web text)
    ↓
SFT (supervised fine-tuning on instruction-response pairs)
    ↓
Preference alignment (DPO / RLHF / GRPO)
    ↓
Safety (red-teaming, constitutional AI, safety classifiers)
    ↓
Evaluation (human eval, benchmarks, A/B testing)
    ↓
Deployment (monitoring, feedback collection, retraining)
```

### Each Stage's Purpose

| Stage | Purpose | Data |
|---|---|---|
| Pretraining | World knowledge, language | Trillions of tokens |
| SFT | Instruction following, format | 10K-1M examples |
| DPO/RLHF | Preference alignment, helpfulness | 10K-100K preferences |
| Safety | Harmlessness, refusal | Red-team prompts |
| Evaluation | Quality assurance | Benchmarks + human eval |

---

## L5 Interview Q&A

### Q: "Walk me through the RLHF pipeline step by step."

1. **SFT**: fine-tune pretrained model on high-quality instruction-response pairs.
2. **Reward model training**: collect human preference pairs (A > B), train a model to predict which response is preferred. The reward model takes (prompt, response) → scalar score.
3. **PPO optimization**: use the reward model to score policy outputs. Optimize policy to maximize reward while staying close to the SFT model (KL penalty).
4. **Evaluation**: check if human raters prefer the RLHF model over the SFT model.

Key components: policy (being trained), reference (frozen SFT), reward model (frozen after training), value model (for PPO advantage estimation).

### Q: "Why does DPO not need a reward model?"

DPO derives a closed-form relationship between the reward and the policy:

```
r(x, y) = β * log(π(y|x) / π_ref(y|x)) + const
```

This means the optimal policy can be expressed directly in terms of the reference model — no need to learn a separate reward model. DPO optimizes the policy directly from preference data using this relationship.

### Q: "How does GRPO differ from PPO for reasoning models?"

PPO needs a value model to estimate advantages (expected future reward). For reasoning tasks (math, code), you can verify correctness automatically — so GRPO samples multiple responses per prompt, scores them, and uses the group statistics (mean, std) as the advantage. No value model needed.

This is why DeepSeek R1 uses GRPO — it's simpler, uses less memory, and works well when rewards are verifiable.

### Q: "What causes reward hacking and how do you detect it?"

**Causes**: reward model has spurious correlations (length, format) that the policy exploits.
**Detection**: 
- Track reward vs human eval — if reward increases but human quality doesn't, you're being hacked.
- Monitor response length, diversity, repetition metrics.
- Use held-out reward model to check for overfitting.

**Mitigations**: KL penalty, length normalization, reward model ensembles, rule-based rewards.

### Q: "When would you use RLAIF vs RLHF?"

- **RLAIF**: when you need scale (millions of preferences), have a strong AI evaluator, and want to reduce cost. Risk: AI evaluator may have blind spots.
- **RLHF**: when quality is critical and you need human judgment. Risk: expensive, slow, inconsistent.
- **Hybrid**: use RLAIF for bulk data, RLHF for quality-critical subsets.

### Q: "Design an alignment pipeline for a customer-facing chatbot."

1. **SFT**: fine-tune on 100K high-quality conversation examples.
2. **DPO**: collect 50K preference pairs from human annotators, train with DPO (simpler than RLHF).
3. **Safety**: red-team with adversarial prompts, add safety examples to DPO data.
4. **Constitutional AI**: use RLAIF to generate additional safety preference data at scale.
5. **Evaluation**: A/B test against current model, track human satisfaction metrics.
6. **Monitoring**: log user feedback (thumbs up/down), alert on safety violations, retrain monthly.

---

## Interview Sound Bites

- RLHF: reward model + PPO + KL penalty — maximum quality, maximum complexity.
- DPO: no reward model, no RL loop — directly optimize from preference pairs using closed-form reward.
- **GRPO** (DeepSeek R1): group-relative advantages, no value model, rule-based rewards for reasoning.
- **SimPO**: no reference model either — simplest and most memory-efficient.
- **KTO**: uses unpaired binary feedback instead of preference pairs — easier data collection.
- **Reward hacking**: policy exploits reward model spurious correlations — mitigate with KL penalty, length normalization, ensembles.
- **RLAIF**: replace human feedback with AI feedback — scalable but may have blind spots.
- Alignment pipeline: Pretraining → SFT → DPO/RLHF → Safety → Evaluation → Deployment.
- DPO is preferred at most labs due to simplicity and competitive quality vs RLHF.
- For reasoning models, GRPO with rule-based rewards avoids reward hacking entirely.
