"""RLHF loss functions in PyTorch: PPO, DPO, and GRPO.

1. PPO (Proximal Policy Optimization):
   Used in RLHF to fine-tune the policy (LLM) using reward model feedback.
   Clips the policy update ratio to prevent destructive large steps.

2. DPO (Direct Preference Optimization):
   Skips the reward model and RL loop entirely.
   Directly optimizes the policy on preference data (chosen vs rejected).
   Simpler, more stable, and often matches RLHF quality.

3. GRPO (Group Relative Policy Optimization):
   Variant of PPO that normalizes rewards within a group of samples.
   Eliminates the need for a separate value/critic model.
   Used in DeepSeek-R1.
"""

import torch


def ppo_loss(new_log_prob, old_log_prob, advantages, eps=0.2):
    """PPO clipped objective (to be maximized, so we negate for loss).

    The ratio r = exp(new_log_prob - old_log_prob) measures how much
    the policy changed. We clip it to [1-eps, 1+eps] to prevent
    destructively large updates.

    Args:
        new_log_prob: log prob under current policy, shape (B,)
        old_log_prob: log prob under old policy (before update), shape (B,)
        advantages:  advantage estimates from reward model + critic, shape (B,)
        eps:         clip range (typically 0.1-0.2).

    Returns:
        Scalar loss (negated PPO objective, for gradient descent).
    """
    # Policy ratio: how much the policy changed relative to old policy.
    ratio = torch.exp(new_log_prob - old_log_prob)

    # Unclipped objective.
    loss1 = ratio * advantages

    # Clipped objective: ratio is bounded to [1-eps, 1+eps].
    loss2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages

    # PPO takes the minimum (pessimistic bound) → conservative updates.
    # Negate because we minimize loss = maximize objective.
    return -torch.min(loss1, loss2).mean()


def dpo_loss(chosen_logp, rejected_logp, beta=0.1):
    """DPO loss for preference optimization.

    DPO directly optimizes the policy to prefer chosen over rejected responses,
    without training a separate reward model or using RL.

    The loss is the negative log of the Bradley-Terry preference probability:
        P(chosen > rejected) = sigmoid(beta * (log_pi(chosen) - log_pi(rejected)))

    Args:
        chosen_logp:   log prob of chosen response under policy, shape (B,)
        rejected_logp: log prob of rejected response under policy, shape (B,)
        beta:          regularization strength (controls how far from reference policy).

    Returns:
        Scalar loss.
    """
    # Preference probability: higher when chosen has higher log prob than rejected.
    probs = torch.sigmoid(beta * (chosen_logp - rejected_logp))

    # Negative log likelihood (we want P(chosen > rejected) → 1).
    return -torch.log(probs).mean()


def grpo_loss(old_logp, new_logp, rewards, eps=0.2):
    """GRPO (Group Relative Policy Optimization) loss.

    Instead of using a critic to estimate advantages, GRPO normalizes
    rewards within a group of samples (e.g., multiple responses to the same prompt).

    Advantage = (reward - group_mean) / group_std

    This eliminates the need for a value model, reducing memory and complexity.

    Args:
        old_logp: log prob under old policy, shape (B,)
        new_logp: log prob under current policy, shape (B,)
        rewards:  raw rewards for the group, shape (B,)
        eps:      PPO clip range.

    Returns:
        Scalar loss.
    """
    # Normalize rewards within the group → advantages.
    group_mean = rewards.mean()
    group_std = rewards.std() + 1e-8  # avoid division by zero
    advantages = (rewards - group_mean) / group_std

    # PPO clipped objective with group-normalized advantages.
    ratio = torch.exp(new_logp - old_logp)
    loss1 = ratio * advantages
    loss2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages

    return -torch.min(loss1, loss2).mean()
