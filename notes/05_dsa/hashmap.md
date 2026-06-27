# HashMap / Dictionary

## When to Use

**Keywords:** `frequency`, `duplicate`, `count`, `unique`, `lookup`, `map`, `seen before`, `group by`, `anagram`

You need to:
- Count occurrences of elements
- Check if something was seen before
- O(1) lookup for existence
- Group items by a key

## Classic Problems
- Two Sum (unsorted)
- Count token frequencies
- First unique character
- Anagram check / Group anagrams
- First duplicate element

## Template — Frequency Count

```python
from collections import defaultdict

def count_frequency(arr):
    freq = defaultdict(int)
    for item in arr:
        freq[item] += 1
    return freq
```

## Template — Seen Before (Duplicate Detection)

```python
seen = set()
for item in items:
    if item in seen:
        return item  # first duplicate
    seen.add(item)
```

## Template — Group By

```python
from collections import defaultdict

groups = defaultdict(list)
for item in items:
    groups[item.key].append(item)
```

## Complexity

| Operation | Average | Worst |
|---|---|---|
| Insert | O(1) | O(N) |
| Lookup | O(1) | O(N) |
| Delete | O(1) | O(N) |

Space: O(U) where U = unique entries
