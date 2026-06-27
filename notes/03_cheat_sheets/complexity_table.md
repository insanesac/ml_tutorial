# Complexity Cheat Sheet

| Pattern | Time | Space | Notes |
|---|---|---|---|
| HashMap insert/lookup/delete | O(1) | O(U) | U = unique entries |
| Set insert/lookup | O(1) | O(U) | |
| Heap push | O(log K) | O(K) | |
| Heap top-K total | O(N log K) | O(K) | Keep heap size = K |
| Binary Search | O(log N) | O(1) | Requires sorted data |
| Sliding Window | O(N) | O(K) | K = window state |
| DFS (time) | O(N) | — | Every node visited once |
| DFS (space, balanced) | — | O(log N) | Recursion stack |
| DFS (space, skewed) | — | O(N) | Recursion stack |
| BFS | O(V + E) | O(V) | |
| Two Pointers | O(N) | O(1) | |
| Sort | O(N log N) | O(N) | |
| Intervals (merge) | O(N log N) | O(N) | Dominated by sort |
| Cosine Similarity | O(D) | O(1) | D = embedding dim |
| Top-K Retrieval | O(ND + N log K) | O(K) | N = docs, D = dim |
| Chunking | O(N) | O(N) | |
| Context Truncation | O(N) | O(N) | |
| Near-Duplicate (pairwise) | O(N²D) | O(1) | All pairs comparison |
