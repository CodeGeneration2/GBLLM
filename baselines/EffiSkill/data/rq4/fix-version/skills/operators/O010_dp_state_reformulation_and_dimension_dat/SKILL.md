---
skill_id: O010
type: operator
language: python
family: state_compression
name: DP State Reformulation and Dimension/Data Structure Optimization
description: Refactor dynamic programs by shrinking or reshaping the state space and replacing naive nested scans with more
  suitable data structures. Typical applications include collapsing unnecessary DP dimensions (e.g., 2D→1D for knapsack-like
  DP, 2-row→1-row for subset DP), pruning unreachable or unneeded states, and turning expensive ‘scan-all-previous’ transitions
  into range queries via segment trees or similar
tags:
- dynamic_programming
- state_reformulation
- dimension_reduction
- space_optimization
- constant_factor_optimization
- unbounded_knapsack
- subset_dp
- bitmask_dp
- set_cover
- digit_dp
triggers:
- A DP table dimension is multiplied by an item index or step index even though transitions only depend on the current and/or
  previous row (e.g., (items+1) x (capacity+1) for unbounded knapsack).
- A 1D knapsack-like recurrence `dp[x] = min(dp[x], dp[x - w_i] + c_i)` or `dp[x] = min_i dp[x - len_i] + cost_i` is implemented
  with an extra DP dimension for items or with a 2D table.
- The outer DP loop iterates over a large extended range (e.g., up to target + max_length) and treats overshoot states both
  as sources and destinations, while the answer only depends on states `>= target`.
- DP transitions involve value constraints like `|a[i] - a[j]| <= D` combined with index constraints (`j` in a sliding window),
  but are implemented by brute-force scanning over candidate indices for each `i`.
- Use of `max( )` or `min( )` over a Python generator/list comprehension in the inner DP loop, especially when the comprehension’s
  range scales with N or a window parameter.
- A huge fixed-sized buffer (e.g., size ~10^6 or more) is allocated for DP regardless of the actual N, sometimes combined
  with tricks like negative indexing or padding inputs.
- Frequent calls to `bin(mask).count('1')` or explicit grouping of masks by popcount inside DP loops, indicating subset-partition
  or convolution-style enumeration rather than straightforward item-by-item transitions.
- Bitmask DP for coverage/set-cover-like problems maintains multiple DP layers (e.g., `dp[2][1<<n]`) and performs both 'keep'
  and 'take' transitions explicitly for every (item, mask) pair.
---

## When to use
- A DP table dimension is multiplied by an item index or step index even though transitions only depend on the current and/or previous row (e.g., (items+1) x (capacity+1) for unbounded knapsack).
- A 1D knapsack-like recurrence `dp[x] = min(dp[x], dp[x - w_i] + c_i)` or `dp[x] = min_i dp[x - len_i] + cost_i` is implemented with an extra DP dimension for items or with a 2D table.
- The outer DP loop iterates over a large extended range (e.g., up to target + max_length) and treats overshoot states both as sources and destinations, while the answer only depends on states `>= target`.
- DP transitions involve value constraints like `|a[i] - a[j]| <= D` combined with index constraints (`j` in a sliding window), but are implemented by brute-force scanning over candidate indices for each `i`.
- Use of `max( )` or `min( )` over a Python generator/list comprehension in the inner DP loop, especially when the comprehension’s range scales with N or a window parameter.
- A huge fixed-sized buffer (e.g., size ~10^6 or more) is allocated for DP regardless of the actual N, sometimes combined with tricks like negative indexing or padding inputs.
- Frequent calls to `bin(mask).count('1')` or explicit grouping of masks by popcount inside DP loops, indicating subset-partition or convolution-style enumeration rather than straightforward item-by-item transitions.
- Bitmask DP for coverage/set-cover-like problems maintains multiple DP layers (e.g., `dp[2][1<<n]`) and performs both 'keep' and 'take' transitions explicitly for every (item, mask) pair.

## Steps
1. Identify the true minimal state needed for the recurrence: For knapsack-like DP, check if the item dimension is only used row-to-row; if so, express the state as a single 1D array over capacity/amount.
2. Identify the true minimal state needed for the recurrence: For subset/bitmask DP, recognize that state can often be `dp[mask]` without an explicit item dimension, updated item-by-item.
3. Identify the true minimal state needed for the recurrence: For digit DP, try to re-express states in terms of (prefix_value, count) or (n, k) using arithmetic on prefixes instead of explicit (position, digit, tight) tables.
4. Collapse unnecessary DP dimensions: Replace 2D knapsack tables `(items+1) x (capacity+1)` with 1D `dp[capacity]`, using correct iteration order for bounded vs unbounded transitions.
5. Collapse unnecessary DP dimensions: Convert 2-layer subset DP `dp[2][1<<n]` into a single `dp[1<<n]` array by interpreting the existing value as the 'skip' case and only explicitly writing 'take' transitions.
6. Collapse unnecessary DP dimensions: For 2-row rolling DPs, maintain just previous and current rows, or merge into one row if transitions allow in-place updates.
7. Prune the DP state space and loops: Limit source indices to those strictly needed (e.g., `for x in range(target)` as sources and treat `x >= target` only as destinations in forward relaxation DPs).
8. Prune the DP state space and loops: Add reachability checks: skip transitions from states where `dp[state]` is still INF/uninitialized.

## Complexity
- Time: Typically reduces from O(N*W) or O(N^2) DP with Python-level nested loops to O(N log M) using range-query data structures, from O(M * 3^N) or O(4^N) subset-partition DPs to O(M * 2^N) item-wise subset DP, and from O(D * K * 10) digit DP
- Space: (pattern dependent)

## Pitfalls
- Incorrect update order when switching from 2D to 1D knapsack DP (e.g., iterating capacity in the wrong direction and effectively turning a bounded problem into an unbounded one, or vice versa).
- For forward-relaxation DPs (e.g., `dp[pos + len]` from `dp[pos]`), accidentally allowing states beyond the target to propagate as new sources, reintroducing the extra iterations the optimization was meant to avoid.
- Mismanaging sliding windows in combination with segment trees: forgetting to remove expired indices or incorrectly clearing values, which can either undercount or overcount valid predecessors.
- Off-by-one errors in range queries and coordinate compression (e.g., mixing inclusive/exclusive ranges, or forgetting to clamp L/R to the domain bounds).
- Using a segment tree where a simpler structure (like a deque for pure monotone windows) would suffice, leading to unnecessary complexity and potential bugs in lazy propagation logic.
- Over-optimizing subset DP by updating a single 1D DP array in the wrong direction, accidentally allowing an item to be applied multiple times when only 0/1 usage is valid.
- Retaining INF states in subset DP transitions: failing to skip masks with INF and still performing arithmetic on them can overflow sentinels or waste work.
- Digit DP refactor errors: mishandling base cases for small n (e.g., n < 10), miscounting numbers equal to N versus strictly less than N, or double-counting/extending off-by-one in the final answer.

## When not to use
- When the original DP state and transitions are already within comfortable time and memory limits, and the added complexity of data structures (segment trees, lazy propagation) increases implementation risk without
- When the dependency structure fundamentally requires full history or 2D relationships (e.g., transitions that depend on arbitrary previous (i,j) states, not compressible to 1D or range aggregates).
- When N and the state space are very small (e.g., N ≤ 100, 2^n tiny), where straightforward O(N^2) or O(M*2^N) solutions are clearer and fast enough, and optimization would only obscure the solution.
- When problem constraints forbid additional logarithmic factors or memory overhead from structures like segment trees, and simpler monotonic-queue/stack or direct DP suffices.
- When using a compressed or reformulated DP would make correctness reasoning significantly more difficult (e.g., in problems with intricate state invariants), increasing the risk of subtle bugs more than it helps
- When the DP recurrence is not monotone or does not decompose into range-aggregate operations, so segment trees or Fenwick trees cannot naturally represent the required transitions.

## Minimal example
Before:
```py
def min_coins(amount, coins):
    n = len(coins)
    INF = 10**9
    dp = [[INF]*(amount+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(1, n+1):
        for x in range(amount+1):
            dp[i][x] = dp[i-1][x]
            if x >= coins[i-1]:  # unbounded usage
                dp[i][x] = min(dp[i][x], dp[i][x - coins[i-1]] + 1)
    return dp[n][amount] if dp[n][amount] < INF else -1
```
After:
```py
def min_coins(amount, coins):
    INF = 10**9
    dp = [INF]*(amount+1)
    dp[0] = 0
    for c in coins:             # collapse item dimension
        for x in range(c, amount+1):  # 1D unbounded knapsack
            if dp[x - c] + 1 < dp[x]:
                dp[x] = dp[x - c] + 1
    return dp[amount] if dp[amount] < INF else -1
```
