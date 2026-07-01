# Heap

## When to Use

**Keywords:** `top k`, `largest k`, `smallest k`, `ranking`, `priority`, `kth`, `nearest`

You need to:
- Find the top/bottom K elements
- Maintain a running priority
- Efficient min/max extraction
- K is much smaller than N

## Min-Heap vs Max-Heap

| Need | Heap Type | Python |
|---|---|---|
| Top K largest | Min-heap of size K | `heapq` (keep smallest at top, evict when > K) |
| Top K smallest | Max-heap of size K | negate values, use `heapq` |
| Kth largest | Min-heap of size K | `heapq` |

## Classic Problems
- Top K frequent elements
- Kth largest element in array
- Merge K sorted lists
- K closest points to origin
- Top K retrieval results

## Template — Top K Largest

```python
import heapq

def top_k_largest(arr, k):
    min_heap = []
    for num in arr:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap  # contains K largest elements
```

## Template — Top K by Score (Retrieval)

```python
import heapq

heap = []
for doc_id, score in scores:
    heapq.heappush(heap, (score, doc_id))
    if len(heap) > k:
        heapq.heappop(heap)
```

## Complexity

| Operation | Time |
|---|---|
| Push | O(log K) |
| Pop | O(log K) |
| Top-K total | O(N log K) |

vs sorting: O(N log N)

Space: O(K)
