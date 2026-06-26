"""Cosine similarity and retrieval in NumPy.

Cosine similarity measures the angle between two vectors, ignoring magnitude:
    cos(a, b) = (a . b) / (||a|| * ||b||)

Range: [-1, 1]
- 1 = same direction (semantically identical)
- 0 = orthogonal (unrelated)
- -1 = opposite direction

Why cosine similarity for embeddings?
- Embeddings encode semantic meaning in direction, not magnitude.
- Two embeddings of the same concept should point in the same direction,
  even if their magnitudes differ.

Retrieval:
- Given a query embedding and a document embedding matrix,
- find the top-k most similar documents by cosine similarity.
"""

import numpy as np


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    Args:
        a: vector, shape (D,) or (..., D)
        b: vector, shape (D,) or (..., D)

    Returns:
        Cosine similarity scalar or array.
    """
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)

    # Guard against zero vectors (would produce NaN).
    if np.any(norm_a == 0) or np.any(norm_b == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return np.dot(a, b) / (norm_a * norm_b)


def retrieval(query, documents, k):
    """Retrieve top-k documents most similar to query by cosine similarity.

    Args:
        query:     query embedding, shape (D,)
        documents: document embeddings, shape (N, D)
        k:         number of top documents to return.

    Returns:
        Indices of top-k most similar documents, sorted by similarity (descending).
    """
    # Normalize query and documents to unit vectors.
    query = query / (np.linalg.norm(query, axis=-1) + 1e-10)
    documents = documents / (np.linalg.norm(documents, axis=-1, keepdims=True) + 1e-10)

    # Cosine similarity = dot product of normalized vectors.
    # (N, D) @ (D,) → (N,)
    similarity = documents @ query

    # Get top-k indices by similarity score.
    top_k_index = np.argpartition(similarity, -k)[-k:]

    # Sort the top-k by similarity in descending order.
    top_k_index = top_k_index[np.argsort(similarity[top_k_index])][::-1]

    return top_k_index


# --- Demo ---
if __name__ == "__main__":
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"cosine_similarity({a}, {b}) = {cosine_similarity(a, b):.4f}")

    # Edge case: zero vector
    # cosine_similarity(np.zeros(3), np.ones(3))  # raises ValueError

    # Retrieval demo
    query = np.array([1.0, 0.0, 0.0])
    docs = np.array([
        [0.9, 0.1, 0.0],   # similar to query
        [0.0, 1.0, 0.0],   # orthogonal
        [0.8, 0.2, 0.1],   # fairly similar
        [0.0, 0.0, 1.0],   # orthogonal
    ])
    print(f"\nTop-2 retrieved indices: {retrieval(query, docs, k=2)}")
