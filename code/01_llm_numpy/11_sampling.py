"""Decoding strategies for LLM text generation in NumPy.

Strategies:
1. Temperature scaling — controls randomness by sharpening/flattening the distribution.
2. Top-k sampling — restrict to the k most likely tokens.
3. Top-p (nucleus) sampling — restrict to the smallest set of tokens whose cumulative
   probability exceeds p.
4. Retrieval — cosine-similarity search, returns top-k document indices.

Greedy decoding (not shown here) simply picks argmax — no randomness, no diversity.
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax over the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def temperature_sampling(logits, T):
    """Apply temperature scaling to logits and sample.

    T < 1: sharper distribution (more confident, less diverse).
    T = 1: original distribution.
    T > 1: flatter distribution (less confident, more diverse).
    T → 0: equivalent to greedy (argmax).
    T → inf: equivalent to uniform random.

    Args:
        logits: raw model output, shape (C,) or (B, C)
        T:      temperature scalar.

    Returns:
        Probability distribution after temperature scaling.
    """
    return softmax(logits / T)


def top_k_sampling(probs, k):
    """Sample from the top-k most probable tokens.

    Args:
        probs: probability distribution, shape (C,)
        k:     number of top tokens to consider.

    Returns:
        Sampled token index.
    """
    # Get indices of top-k probabilities.
    top_k_index = np.argpartition(probs, -k)[-k:]

    # Renormalize the top-k probabilities to sum to 1.
    top_k_probs = probs[top_k_index] / np.sum(probs[top_k_index])

    # Sample from the restricted distribution.
    return np.random.choice(top_k_index, p=top_k_probs)


def top_p_sampling(probs, p):
    """Sample from the nucleus (top-p) set.

    Nucleus = the smallest set of tokens whose cumulative probability >= p.
    This adapts the number of candidates based on the distribution shape.

    Args:
        probs: probability distribution, shape (C,)
        p:     cumulative probability threshold (e.g., 0.9).

    Returns:
        Sampled token index.
    """
    # Sort probabilities in descending order.
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Compute cumulative probabilities.
    cumsum = np.cumsum(sorted_probs)

    # Find the cutoff index where cumulative probability first exceeds p.
    cutoff = np.searchsorted(cumsum, p) + 1

    # Select the nucleus tokens.
    top_indices = sorted_indices[:cutoff]
    top_p_probs = probs[top_indices]

    # Renormalize and sample.
    top_p_probs = top_p_probs / np.sum(top_p_probs)

    return np.random.choice(top_indices, p=top_p_probs)


def retrieval(doc, query, k):
    """Return indices of the top-k documents most similar to the query.

    Uses cosine similarity (L2-normalise both, then dot product).

    Args:
        doc:   document embeddings, shape (N, d)
        query: query embedding, shape (d,)
        k:     number of top results to return.

    Returns:
        Array of k indices sorted by descending similarity.
    """
    query = query / np.linalg.norm(query, axis=-1)
    doc = doc / np.linalg.norm(doc, axis=-1, keepdims=True)

    similarity = doc @ query

    top_k_indices = np.argpartition(similarity, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(similarity[top_k_indices])][::-1]
    return top_k_indices


# --- Demo ---
if __name__ == "__main__":
    logits = np.array([1.0, 2.0, 3.0, 0.5, 0.1])

    print(f"Temperature T=0.5: {temperature_sampling(logits, 0.5)}")
    print(f"Temperature T=2.0: {temperature_sampling(logits, 2.0)}")

    probs = softmax(logits)
    print(f"\nTop-k=3 sample: {top_k_sampling(probs, k=3)}")
    print(f"Top-p=0.9 sample: {top_p_sampling(probs, p=0.9)}")

    docs = np.random.randn(10, 8)
    query = np.random.randn(8)
    print(f"\nRetrieval top-3 indices: {retrieval(docs, query, k=3)}")
