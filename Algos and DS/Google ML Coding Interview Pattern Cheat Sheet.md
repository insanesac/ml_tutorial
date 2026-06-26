# Google ML Coding Interview Pattern Cheat Sheet

## Goal

For every problem:

```text
Problem
↓
Recognize Pattern
↓
Choose Data Structure
↓
Apply Template
↓
Code
```

---

# 1. HashMap / Dictionary

## When to Use

Keywords:

* Count
* Frequency
* Occurrence
* Duplicate
* Group By
* Lookup
* Anagram

Examples:

* Count token frequencies
* First duplicate element
* Two Sum (unsorted)
* Group Anagrams

---

## Core Operations

```python
counts = {}

counts[key] = counts.get(key, 0) + 1
```

Membership:

```python
if key in counts:
```

Lookup:

```python
counts[key]
```

---

## Complexity

Average:

```text
Insert: O(1)
Lookup: O(1)
Delete: O(1)
```

Space:

```text
O(U)
```

where U = unique entries.

---

## Template

```python
counts = {}

for item in items:
    counts[item] = counts.get(item, 0) + 1
```

---

# 2. Set

## When to Use

Keywords:

* Seen before?
* Duplicate detection
* Unique elements
* Membership testing

Examples:

* First duplicate token
* Longest unique substring
* Deduplication

---

## Template

```python
seen = set()

for item in items:

    if item in seen:
        return item

    seen.add(item)
```

---

## Complexity

```text
Insert: O(1)
Lookup: O(1)
Delete: O(1)
```

---

# 3. Heap

## When to Use

Keywords:

* Top K
* Largest K
* Smallest K
* Ranking
* Priority

Examples:

* Top K frequent tokens
* Top K retrieval results
* Top K recommendations

---

## Python

```python
import heapq
```

Python provides:

```text
Min Heap
```

---

## Push

```python
heapq.heappush(heap, value)
```

---

## Pop

```python
heapq.heappop(heap)
```

Removes smallest element.

---

## Top K Pattern

Maintain heap size K.

```python
for item in items:

    heapq.heappush(heap, item)

    if len(heap) > k:
        heapq.heappop(heap)
```

---

## Complexity

```text
O(U log K)
```

vs sorting:

```text
O(U log U)
```

---

# 4. Sliding Window

Used for contiguous ranges.

Keywords:

* Subarray
* Substring
* Longest
* Shortest
* Continuous
* Window

---

# 4A. Fixed Sliding Window

## When

Window size known.

Examples:

* Moving average over 7 days
* Maximum sum of size K

---

## Template

```python
window_sum = sum(nums[:k])

for right in range(k, len(nums)):

    window_sum -= nums[right-k]
    window_sum += nums[right]

    update_answer()
```

---

## Complexity

```text
O(N)
```

---

# 4B. Dynamic Sliding Window

## When

Window size changes.

Examples:

* Longest unique substring
* Shortest window containing keywords
* Longest subarray under threshold

---

## Template

```python
left = 0

for right in range(len(arr)):

    expand_window()

    while invalid:
        shrink_window()

    update_answer()
```

---

## Key Insight

Same template.

Only the validity condition changes.

---

# 5. Two Pointers

## When

Keywords:

* Sorted array
* Pair sum
* Opposite ends

---

## Example

```python
nums = [1,2,4,6,8,10]
target = 12
```

---

## Template

```python
left = 0
right = len(nums)-1

while left < right:

    current = nums[left] + nums[right]

    if current == target:
        return [left, right]

    elif current < target:
        left += 1

    else:
        right -= 1
```

---

## Complexity

```text
O(N)
```

---

# 6. Binary Search

## When

Keywords:

* Sorted
* First occurrence
* Last occurrence
* Greater than
* Threshold
* Minimum value satisfying condition

---

## Template

```python
left = 0
right = len(nums)-1

while left <= right:

    mid = (left + right) // 2

    if nums[mid] == target:
        return mid

    elif nums[mid] < target:
        left = mid + 1

    else:
        right = mid - 1
```

---

## Complexity

```text
O(log N)
```

---

# 7. DFS

## Think

Go Deep.

---

## When

Keywords:

* Tree
* Depth
* Path
* Hierarchy

Examples:

* Maximum depth
* Tree traversal

---

## Template

```python
def dfs(node):

    if not node:
        return

    dfs(node.left)
    dfs(node.right)
```

---

## Complexity

```text
O(N)
```

---

# 8. BFS

## Think

Level by Level.

---

## When

Keywords:

* Level order
* Distance
* Shortest path
* Breadth

---

## Template

```python
from collections import deque

q = deque([root])

while q:

    node = q.popleft()

    if node.left:
        q.append(node.left)

    if node.right:
        q.append(node.right)
```

---

## Complexity

```text
O(N)
```

---

# 9. Intervals

## When

Keywords:

* Meeting rooms
* Calendar
* Schedule
* Overlap
* Time range

---

## Example

```python
[
 [1,3],
 [2,6],
 [8,10]
]
```

---

## Overlap Rule

```python
start2 <= end1
```

---

## Merge Rule

```python
[
 min(start1,start2),
 max(end1,end2)
]
```

---

## Template

```python
intervals.sort()

merged = []

for interval in intervals:

    if not merged:
        merged.append(interval)

    elif merged[-1][1] >= interval[0]:

        merged[-1][1] = max(
            merged[-1][1],
            interval[1]
        )

    else:
        merged.append(interval)
```

---

# Pattern Recognition Table

| Keywords                    | Pattern        |
| --------------------------- | -------------- |
| frequency, count, duplicate | HashMap        |
| unique, seen before         | Set            |
| top k, ranking              | Heap           |
| longest substring           | Sliding Window |
| shortest subarray           | Sliding Window |
| sorted pair sum             | Two Pointers   |
| sorted + first/last         | Binary Search  |
| tree depth                  | DFS            |
| level order                 | BFS            |
| overlap, meetings           | Intervals      |

---

# Interview Strategy

1. Clarify requirements.
2. State brute force.
3. Analyze complexity.
4. Identify pattern.
5. Explain optimized solution.
6. Code.
7. Test edge cases.
8. State final complexity.
