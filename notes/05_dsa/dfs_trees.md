# DFS on Trees: The Mental Model

## The Biggest Mistake Beginners Make

When solving tree problems, most people think:

> "I need to traverse the tree."

That's not quite right. A better way to think is:

> "Every node asks its children for information."

The entire DFS pattern is built around this idea.

---

## Tree Representation

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
```

Example tree:

```
      1
     / \
    2   3
   / \
  4   5
```

`node.left` points to another `TreeNode`. `node.right` points to another `TreeNode`. The picture is simply a visualization of these pointers.

---

## What is a Subtree?

A subtree is a node and everything below it.

```
Whole tree:          Subtree at 2:       Subtree at 4:
      1                  2                  4
     / \                / \
    2   3              4   5
   / \
  4   5
```

---

## The DFS Recipe

Every DFS problem follows the same structure.

### Step 1: Decide what information you want each child to return

| Problem | Child returns |
|---|---|
| Max Depth | Depth of its subtree |
| Count Nodes | Number of nodes in its subtree |
| Tree Sum | Sum of all values in its subtree |
| Maximum Value | Maximum value inside its subtree |
| Search | True / False |

### Step 2: Base Case

Ask: "What should an empty tree return?"

| Problem | `None` returns |
|---|---|
| Max Depth | 0 |
| Count Nodes | 0 |
| Tree Sum | 0 |
| Search | False |

### Step 3: Ask Both Children

```python
left_answer = solve(left subtree)
right_answer = solve(right subtree)
```

Not actual values — the COMPLETE answer for that subtree.

### Step 4: Combine

This is the only part that changes from problem to problem.

---

## Worked Examples

### Maximum Depth

```
Node 4 (leaf): returns 1  (1 + max(0, 0))
Node 2: left_depth=1, right_depth=1 → returns 1 + max(1, 1) = 2
```

```python
return 1 + max(left_depth, right_depth)
```

### Count Nodes

```
Node 2: left_count=1, right_count=1 → returns 1 + 1 + 1 = 3
```

```python
return 1 + left_count + right_count
```

### Tree Sum

```
Node 2: left_sum=4, right_sum=5 → returns 2 + 4 + 5 = 11
```

```python
return node.val + left_sum + right_sum
```

### Maximum Value

```
Node 5: left_max=100, right_max=3 → returns max(5, 100, 3) = 100
```

```python
return max(node.val, left_max, right_max)
```

### Search

```
Children return True/False
```

```python
return node.val == target or found_left or found_right
```

---

## Notice the Pattern

Every DFS problem has identical structure. Only the combine step changes.

```
Base Case → Ask Left Child → Ask Right Child → Combine Answers → Return Upward
```

### Generic DFS Template

```python
def dfs(node):
    if node is None:
        return BASE_CASE
    left = dfs(node.left)
    right = dfs(node.right)
    return COMBINE(node, left, right)
```

Everything else is just replacing `BASE_CASE` and `COMBINE`.

---

## Complexity

| | Time | Space |
|---|---|---|
| Balanced tree | O(N) | O(log N) |
| Skewed tree | O(N) | O(N) |

Time: every node visited exactly once. Space: recursion stack depth.

---

## Interview Pattern Recognition

If interviewer says: **height, depth, count, sum, maximum, minimum, search, path sum, diameter**

Immediately think: **DFS**

Then ask yourself:
1. What should `None` return?
2. What should my children tell me?
3. How do I combine their answers?

If you can answer those three questions, you've essentially solved the DFS problem.

---

## Key Insight

Don't think: *"I'm traversing the tree."*

Think: *"Each node asks its children for the answer, combines those answers with its own information, and returns a single result to its parent."*

That mental model works for the vast majority of recursive tree problems.
