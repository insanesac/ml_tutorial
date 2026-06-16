# RLHF & DPO (Alignment)

## Why Alignment Exists

A pretrained LLM is trained to predict the next token.

That does not mean it is helpful, harmless, or honest.

Alignment techniques fine-tune the model to follow instructions and match human preferences.

---

## RLHF (Reinforcement Learning from Human Feedback)

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

---

## DPO (Direct Preference Optimization)

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

## Interview Sound Bites

- RLHF uses a reward model + PPO to align with human preferences.
- DPO eliminates the reward model and RL loop, optimizing preference data directly.
- Both use a reference model to prevent the policy from drifting too far.
- KL penalty is what stops reward hacking.
- DPO is now preferred at most labs due to simplicity and competitive quality.
