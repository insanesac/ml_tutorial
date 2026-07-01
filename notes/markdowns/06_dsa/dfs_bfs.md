# DFS / BFS (Trees & Graphs)

## When to Use

**Keywords:** `tree`, `hierarchy`, `traverse`, `path`, `connected`, `level`, `depth`, `adjacency`, `graph`

| Pattern | Use When |
|---|---|
| **DFS** | Explore deep paths, backtracking, topological sort, tree traversal |
| **BFS** | Shortest path (unweighted), level-order traversal, nearest neighbors |

## DFS Variants

| Variant | Description |
|---|---|
| **Pre-order** | Root → Left → Right |
| **In-order** | Left → Root → Right (gives sorted order in BST) |
| **Post-order** | Left → Right → Root (useful for deletion, subtree computation) |

## Classic Problems
- Tree traversals (inorder, preorder, postorder)
- Maximum depth of tree
- Level order traversal (BFS)
- Number of islands (DFS/BFS on grid)
- Word ladder (BFS shortest path)
- Course schedule (topological sort)

## Template — DFS on Tree

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

## Template — BFS Level Order

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

## Complexity

Time: O(V + E) where V = vertices, E = edges
Space: O(V) for visited set / queue / recursion stack
