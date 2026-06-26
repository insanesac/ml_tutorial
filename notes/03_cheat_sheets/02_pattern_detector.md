# The Pattern Detector — DSA Keyword to Pattern Map

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

## 1. HashMap

### Trigger Keywords
`frequency`, `duplicate`, `count`, `unique`, `lookup`, `map`, `seen before`

### When to Use
- You need to count occurrences of elements
- You need to check if something was seen before
- You need O(1) lookup for existence
- You need to group items by a key

### Classic Problems
- Two Sum
- Frequency counter
- First unique character
- Anagram check
- Group anagrams

### Template

```python
from collections import defaultdict

def count_frequency(arr):
    freq = defaultdict(int)
    for item in arr:
        freq[item] += 1
    return freq
```

### Complexity
- Time: O(N)
- Space: O(N)

---

## 2. Heap

### Trigger Keywords
`top k`, `largest k`, `smallest k`, `ranking`, `priority`, `kth`, `nearest`

### When to Use
- You need the top/bottom K elements
- You need to maintain a running priority
- You need efficient min/max extraction
- K is much smaller than N

### Min-Heap vs Max-Heap

| Need | Heap Type | Python |
|---|---|---|
| Top K largest | Min-heap of size K | `heapq` (keep smallest at top, evict when > K) |
| Top K smallest | Max-heap of size K | negate values, use `heapq` |
| Kth largest | Min-heap of size K | `heapq` |

### Classic Problems
- Top K frequent elements
- Kth largest element in array
- Merge K sorted lists
- K closest points to origin

### Template

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

### Complexity
- Time: O(N log K)
- Space: O(K)

---

## 3. Sliding Window

### Trigger Keywords
`substring`, `subarray`, `longest`, `shortest`, `consecutive`, `contiguous`, `window`

### When to Use
- You need to find a contiguous subarray/substring
- You need to optimize over a range of fixed or variable size
- You can expand/contract a window rather than rescanning

### Fixed vs Variable Window

| Type | When | Example |
|---|---|---|
| **Fixed window** | Window size is given | "Max sum subarray of size K" |
| **Variable window** | Window size is unknown | "Longest substring without repeating chars" |

### Classic Problems
- Longest substring without repeating characters
- Maximum sum subarray of size K
- Minimum window substring
- Longest substring with at most K distinct characters

### Template (Variable Window)

```python
def sliding_window(s):
    left = 0
    result = 0
    window_state = {}  # or set, counter, etc.

    for right in range(len(s)):
        # expand: add s[right] to window
        window_state[s[right]] = window_state.get(s[right], 0) + 1

        # contract: shrink from left while condition violated
        while condition_violated(window_state):
            window_state[s[left]] -= 1
            if window_state[s[left]] == 0:
                del window_state[s[left]]
            left += 1

        # update result
        result = max(result, right - left + 1)

    return result
```

### Complexity
- Time: O(N) — each element added once, removed once
- Space: O(K) where K is the window state size

---

## 4. Binary Search

### Trigger Keywords
`sorted array`, `sorted`, `threshold`, `first occurrence`, `last occurrence`, `find position`, `O(log n)`, `monotonic`

### When to Use
- Input is sorted (or monotonic)
- You need to find a position, threshold, or boundary
- You need better than O(N)
- The search space can be halved each step

### Variants

| Variant | Description |
|---|---|
| **Standard** | Find exact match in sorted array |
| **Lower bound** | First position where value >= target |
| **Upper bound** | First position where value > target |
| **Binary search on answer** | Search over possible answers, check feasibility |

### Classic Problems
- Binary search in sorted array
- First and last position of element
- Search insert position
- Capacity to ship packages within D days (binary search on answer)
- Koko eating bananas

### Template

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1  # not found
```

### Template (Find First Occurrence)

```python
def first_occurrence(arr, target):
    lo, hi = 0, len(arr) - 1
    result = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            result = mid
            hi = mid - 1  # keep searching left
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return result
```

### Complexity
- Time: O(log N)
- Space: O(1)

---

## 5. Intervals

### Trigger Keywords
`overlap`, `meetings`, `schedule`, `merge`, `insert`, `intervals`, `ranges`

### When to Use
- You are working with ranges (start, end pairs)
- You need to merge, find overlaps, or insert intervals
- Sorting by start time is usually the first step

### Classic Problems
- Merge intervals
- Insert interval
- Meeting rooms I / II
- Non-overlapping intervals

### Template (Merge Intervals)

```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])  # sort by start
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_end = merged[-1][1]
        if start <= last_end:  # overlap
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])

    return merged
```

### Complexity
- Time: O(N log N) — dominated by sort
- Space: O(N) — for output

---

## 6. DFS / BFS (Trees & Graphs)

### Trigger Keywords
`tree`, `hierarchy`, `traverse`, `path`, `connected`, `level`, `depth`, `adjacency`, `graph`

### When to Use

| Pattern | Use When |
|---|---|
| **DFS** | Explore deep paths, backtracking, topological sort, tree traversal |
| **BFS** | Shortest path (unweighted), level-order traversal, nearest neighbors |

### DFS Variants

| Variant | Description |
|---|---|
| **Pre-order** | Root → Left → Right |
| **In-order** | Left → Root → Right (gives sorted order in BST) |
| **Post-order** | Left → Right → Root (useful for deletion, subtree computation) |

### Classic Problems
- Tree traversals (inorder, preorder, postorder)
- Level order traversal (BFS)
- Number of islands (DFS/BFS on grid)
- Word ladder (BFS shortest path)
- Course schedule (topological sort)

### Template (DFS on Tree)

```python
def dfs(node):
    if not node:
        return

    # process node (pre-order)
    dfs(node.left)
    # process node (in-order)
    dfs(node.right)
    # process node (post-order)
```

### Template (BFS Level Order)

```python
from collections import deque

def bfs_level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

### Complexity
- Time: O(V + E) where V = vertices, E = edges
- Space: O(V) for visited set / queue / recursion stack

---

## 7. Two Pointers

### Trigger Keywords
`pair in sorted array`, `two sum`, `sorted`, `palindrome`, `reverse`, `left right`, `meet in middle`

### When to Use
- Array is sorted (or can be sorted)
- You need to find a pair/triplet satisfying a condition
- You need to compare elements from both ends
- You need to avoid nested loops (O(N^2) → O(N))

### Variants

| Variant | Description |
|---|---|
| **Opposite direction** | One pointer at start, one at end, move toward center |
| **Same direction** | Fast and slow pointers (e.g., cycle detection) |
| **Two arrays** | One pointer per array, merge-style |

### Classic Problems
- Two sum (sorted array)
- Three sum
- Container with most water
- Valid palindrome
- Remove duplicates from sorted array

### Template (Opposite Direction)

```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:
        current = arr[left] + arr[right]

        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1

    return []  # not found
```

### Complexity
- Time: O(N)
- Space: O(1)

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

---

## Interview Discipline

1. **Read the problem twice.**
2. **Identify keywords.**
3. **Map to pattern.**
4. **Say the pattern out loud** before writing code.
5. **If no pattern matches**, consider: dynamic programming, backtracking, or greedy.
6. **If multiple patterns match**, pick the one with better complexity.
