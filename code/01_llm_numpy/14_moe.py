"""Mixture of Experts (MoE) forward pass in NumPy.

MoE routes each token to a subset of experts (FFN modules) rather than
using a single FFN. This increases model capacity without proportional
compute increase.

Key components:
1. Router: linear layer that produces a probability for each expert.
2. Top-k selection: pick the k experts with highest probability.
3. Weighted combination: sum of expert outputs weighted by router probs.

Example with k=2:
- Token → router → [0.1, 0.6, 0.3, 0.05] (4 experts)
- Top-2: experts 1 and 2 (probs 0.6 and 0.3)
- Renormalize: 0.6/0.9 = 0.67, 0.3/0.9 = 0.33
- Output = 0.67 * expert_1(x) + 0.33 * expert_2(x)
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax over the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def moe_forward(x, experts, W_router, k=2):
    """Mixture of Experts forward pass for a single token.

    Args:
        x:        input token, shape (D,)
        experts:  list of expert callables, each: f(x) -> output
        W_router: router weights, shape (D, num_experts)
        k:        number of experts to activate per token.

    Returns:
        Weighted sum of top-k expert outputs, shape (D_out,).
    """
    # Router produces logits for each expert.
    router_logits = x @ W_router  # (num_experts,)

    # Convert to probabilities.
    router_probs = softmax(router_logits)  # (num_experts,)

    # Select top-k experts by probability.
    topk_idx = np.argsort(router_probs)[::-1][:k]  # (k,)

    # Renormalize the top-k probabilities to sum to 1.
    topk_probs = router_probs[topk_idx] / np.sum(router_probs[topk_idx])

    # Weighted sum of selected expert outputs.
    output = 0
    for prob, idx in zip(topk_probs, topk_idx):
        output += prob * experts[idx](x)

    return output


# --- Demo ---
if __name__ == "__main__":
    D = 64
    num_experts = 4

    # Create dummy experts (each is a simple linear layer + ReLU).
    experts = [
        lambda x: np.maximum(0, x @ np.random.randn(D, D))
        for _ in range(num_experts)
    ]

    W_router = np.random.randn(D, num_experts)
    x = np.random.randn(D)

    out = moe_forward(x, experts, W_router, k=2)
    print(f"MoE output shape: {out.shape}")
