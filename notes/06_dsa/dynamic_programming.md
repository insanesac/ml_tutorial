# Dynamic Programming Interview Cheat Sheet

## Definition

Dynamic Programming (DP) solves problems by storing solutions to overlapping subproblems and reusing them.

### Two Key Properties

1. **Overlapping Subproblems** — same subproblems solved repeatedly
2. **Optimal Substructure** — optimal solution can be built from optimal solutions of subproblems

## Common Approaches

| | Memoization (Top-Down) | Tabulation (Bottom-Up) |
|---|---|---|
| Direction | Recursion → Cache → Reuse | Small → Large → Final |
| Style | Recursive with cache | Iterative, fill table |

## Interview Pattern

1. **Define State** — What does `dp[i]` represent?
2. **Recurrence Relation** — Express current state using previous states
3. **Base Case** — Initialize smallest problems
4. **Transition** — Fill remaining states

## Classic Problems

- Fibonacci
- Climbing Stairs
- House Robber
- Coin Change
- Knapsack
- Longest Increasing Subsequence
- Longest Common Subsequence
- Edit Distance
- Partition Equal Subset Sum
- Unique Paths

## Optimization

Many DP problems reduce O(N²) → O(N) by keeping only previous states (space optimization).

## Interview Tips

Always identify **State**, **Transition**, **Base Case** before writing code.

### Google ML Coding

DP appears less frequently than:
- Arrays, Hash Maps, Trees, Graphs, Sliding Window, Heap, Binary Search

A quick review of classic patterns is generally sufficient for ML-focused coding interviews.

## Interview Summary

Dynamic Programming is an optimization technique that solves problems with overlapping subproblems and optimal substructure by storing intermediate results. Successful DP solutions begin by defining the state, recurrence relation, base cases, and transition before implementing either memoization or tabulation.
