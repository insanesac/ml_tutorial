import numpy as np

def cosine_similarity(a, b):
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        raise ValueError(...)
    return (np.dot(a,b))/(np.linalg.norm(b) * np.linalg.norm(b))

#edge case

#[0 0 0]

"""Embeddings encode semantic meaning in direction.

Cosine similarity measures similarity of direction
while ignoring vector magnitude."""