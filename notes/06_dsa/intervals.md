# Intervals

## When to Use

**Keywords:** `overlap`, `meetings`, `schedule`, `merge`, `insert`, `intervals`, `ranges`

Working with ranges (start, end pairs). Sorting by start time is usually the first step.

## Classic Problems
- Merge intervals
- Insert interval
- Meeting rooms I / II
- Non-overlapping intervals

## Overlap Rule

```python
start2 <= end1  # intervals overlap
```

## Merge Rule

```python
[min(start1, start2), max(end1, end2)]
```

## Template — Merge Intervals

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

## Complexity

Time: O(N log N) — dominated by sort
Space: O(N) — for output
