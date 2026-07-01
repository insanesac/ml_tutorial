# Two Pointers

## When to Use

**Keywords:** `pair in sorted array`, `two sum`, `sorted`, `palindrome`, `reverse`, `left right`, `meet in middle`

Array is sorted (or can be sorted), and you need to find a pair/triplet satisfying a condition, or compare elements from both ends.

## Variants

| Variant | Description |
|---|---|
| **Opposite direction** | One pointer at start, one at end, move toward center |
| **Same direction** | Fast and slow pointers (e.g., cycle detection) |
| **Two arrays** | One pointer per array, merge-style |

## Classic Problems
- Two sum (sorted array)
- Three sum
- Container with most water
- Valid palindrome
- Remove duplicates from sorted array

## Template — Opposite Direction (Two Sum Sorted)

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

**Logic:** If sum < target, increase left. If sum > target, decrease right.

## Complexity

Time: O(N)
Space: O(1)
