# Binary Search

## When to Use

**Keywords:** `sorted array`, `sorted`, `threshold`, `first occurrence`, `last occurrence`, `find position`, `O(log n)`, `monotonic`

Input is sorted (or monotonic), and you need to find a position, threshold, or boundary.

## Variants

| Variant | Description |
|---|---|
| **Standard** | Find exact match in sorted array |
| **Lower bound** | First position where value >= target |
| **Upper bound** | First position where value > target |
| **Binary search on answer** | Search over possible answers, check feasibility |

## Classic Problems
- Binary search in sorted array
- First and last position of element
- Search insert position
- Capacity to ship packages within D days (binary search on answer)
- Koko eating bananas

## Template — Standard

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

## Template — First Occurrence (Lower Bound)

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

## Template — Upper Bound (First element > target)

```python
def upper_bound(arr, target):
    lo, hi = 0, len(arr) - 1
    result = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] > target:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return result
```

## Complexity

Time: O(log N)
Space: O(1)
