# Pattern Recognition — Keyword to Pattern Map

Train your brain to recognize the pattern before writing any code.

---

## Quick Reference Table

| Keywords | Pattern |
|---|---|
| frequency, duplicate, count | **HashMap** |
| top k, largest k, ranking | **Heap** |
| substring, subarray, longest | **Sliding Window** |
| sorted array, threshold, first/last occurrence | **Binary Search** |
| overlap, meetings, schedule | **Intervals** |
| tree, hierarchy | **DFS / BFS** |
| pair in sorted array | **Two Pointers** |

---

## Pattern Recognition Drill

Read the problem, say the pattern out loud before coding:

| Problem Statement | Pattern | Why |
|---|---|---|
| "Count how many times each word appears" | HashMap | frequency |
| "Find the 3rd largest element" | Heap | top k |
| "Longest substring with at most 2 distinct chars" | Sliding Window | substring + longest |
| "Find first position of 5 in sorted array" | Binary Search | sorted + first occurrence |
| "Merge overlapping meeting times" | Intervals | overlap + meetings |
| "Check if two nodes are connected in a graph" | DFS/BFS | connected + graph |
| "Find two numbers that sum to target in sorted array" | Two Pointers | pair + sorted |
| "Top 5 most similar documents to query" | Cosine Similarity + Heap | semantic search + top k |
| "Split document into chunks of 500 tokens with 50 overlap" | Sliding Window | chunking + overlap |
| "Keep conversation under 4000 tokens" | Greedy Reverse | context management |

---

## Interview Discipline

1. **Read the problem twice.**
2. **Identify keywords.**
3. **Map to pattern.**
4. **Say the pattern out loud** before writing code.
5. **If no pattern matches**, consider: dynamic programming, backtracking, or greedy.
6. **If multiple patterns match**, pick the one with better complexity.

---

## ML Coding Pattern Table

| Problem Type | Pattern |
|---|---|
| Frequency Count | HashMap |
| Deduplication | Set |
| Top K Results | Heap |
| Retrieval | Similarity + Heap |
| Semantic Search | Embedding + Top K |
| Context Management | Sliding Window |
| Chunking | Sliding Window |
| Ranking | Sort / Heap |
| Recall@K | Set Operations |
| MRR | Ranking Scan |
| RAG Retrieval | Similarity + Top K + Rerank |
| Conversation Memory | Sliding Window |
