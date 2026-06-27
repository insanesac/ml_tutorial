# Sliding Window

## When to Use

**Keywords:** `substring`, `subarray`, `longest`, `shortest`, `consecutive`, `contiguous`, `window`

You need to find a contiguous subarray/substring, or optimize over a range.

## Fixed vs Variable Window

| Type | When | Example |
|---|---|---|
| **Fixed window** | Window size is given | "Max sum subarray of size K" |
| **Variable window** | Window size is unknown | "Longest substring without repeating chars" |

## Classic Problems
- Longest substring without repeating characters
- Maximum sum subarray of size K
- Minimum window substring
- Longest substring with at most K distinct characters
- Chunking with overlap

## Template — Fixed Window

```python
window_sum = sum(nums[:k])

for right in range(k, len(nums)):
    window_sum -= nums[right - k]     # remove leftmost
    window_sum += nums[right]         # add new right
    update_answer()
```

## Template — Variable Window

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

## Chunking with Overlap (LLM-specific)

```python
start = 0
while start < len(tokens):
    end = start + chunk_size
    chunk = tokens[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap  # step = chunk_size - overlap
```

**Edge case:** `overlap >= chunk_size` → infinite loop

## Key Insight

Same template. Only the validity condition changes.

## Complexity

Time: O(N) — each element added once, removed once
Space: O(K) where K is the window state size
