---
skill_id: O021
type: operator
language: python
family: streaming
name: Refactor Redundant Range Loops and Pairwise Aggregations
description: Reusable optimization pattern distilled from weighted traces.
tags:
- optimization
- algorithmic
- time-complexity
- range-search
- pairwise-aggregation
- prefix-sums
- math-identity
- convex-optimization-1d
- sign-enumeration
- sorting
triggers:
- Loop over indices where the body calls sum() or max()/min() on (almost) the entire list in each iteration.
- Presence of an outer loop whose induction variable is never used inside the body, yet wraps a full candidate search or aggregation.
- Triple or double nested loops where two loops iterate over similar value ranges (e.g., min..max of data), yielding O(R^2
  * N) work for a scalar parameter search.
- Searching over a scalar parameter p in a data-dependent range [min(a), max(a)] even though the problem guarantees a small
  fixed domain (e.g., values in [1, 100]).
- Greedy selection loops of the form `for k in range(M):` with an inner scan over all N items to recompute a 'best' candidate
  each step (O(N*M) ≈ O(N^2)).
- Recomputing the same global aggregate (sum(a), sum(a^2), sum of squared differences, etc.) from scratch inside a loop where
  the underlying data does not change.
- Using slicing or list.index()/max(list_slice) repeatedly inside a loop over the same list, effectively causing quadratic
  scans.
- Objectives that are sums of squared deviations in a scalar parameter (convex/unimodal in p), but evaluated at every candidate
  in a fixed range without early stopping.
---

## When to use
- Loop over indices where the body calls sum() or max()/min() on (almost) the entire list in each iteration.
- Presence of an outer loop whose induction variable is never used inside the body, yet wraps a full candidate search or aggregation.
- Triple or double nested loops where two loops iterate over similar value ranges (e.g., min..max of data), yielding O(R^2 * N) work for a scalar parameter search.
- Searching over a scalar parameter p in a data-dependent range [min(a), max(a)] even though the problem guarantees a small fixed domain (e.g., values in [1, 100]).
- Greedy selection loops of the form `for k in range(M):` with an inner scan over all N items to recompute a 'best' candidate each step (O(N*M) ≈ O(N^2)).
- Recomputing the same global aggregate (sum(a), sum(a^2), sum of squared differences, etc.) from scratch inside a loop where the underlying data does not change.
- Using slicing or list.index()/max(list_slice) repeatedly inside a loop over the same list, effectively causing quadratic scans.
- Objectives that are sums of squared deviations in a scalar parameter (convex/unimodal in p), but evaluated at every candidate in a fixed range without early stopping.

## Steps
1. Identify the true logical objective: For pairwise products/sums: recognize patterns like Σ_{i<j} a[i]*a[j] or repeated pair interactions.
2. Identify the true logical objective: For 1D scalar searches: recognize objectives of the form Σ f(a_i, p), especially squared deviations.
3. Identify the true logical objective: For absolute-sum geometry objectives: rewrite |Σ coords| as a max over a small set of sign patterns.
4. Remove redundant outer value-range loops: If a loop variable over a candidate range is unused in the body, delete that loop and keep only the inner candidate loop.
5. Remove redundant outer value-range loops: Ensure that each candidate parameter value p is evaluated at most once.
6. Replace O(N^2) aggregations with O(N) math identities: For pairwise products or sums, use closed-form identities such as ((Σ a)^2 − Σ a^2)/2 or prefix-sum formulations instead of explicit double loops or repeated sum(a) calls.
7. Replace O(N^2) aggregations with O(N) math identities: For max difference problems, track running prefix minima/maxima in a single pass instead of recomputing suffix max/min per position.
8. Optimize scalar parameter searches over p: Restrict the candidate domain using problem constraints (e.g., known bounds like [1, 100]) instead of [min(a), max(a)] when valid.

## Complexity
- Time: Typically reduces: (1) O(N^2) pairwise or repeated-scan algorithms to O(N) via math identities or prefix sums; (2) O(R^2 * N) scalar-parameter searches (two nested candidate loops over value ranges) to O(R * N) by eliminating redundant
- Space: Usually O(N) space for storing input and simple auxiliary arrays or prefix sums. Some variants remain O(1) extra space beyond input. When using vectorization or sign-enumeration, additional O(N) arrays per pattern may be created, but the

## Pitfalls
- Leaving an unused outer loop over a value range in place, which repeats the full search R times and leaves complexity at O(R^2 * N).
- Caching a precomputed aggregate (like total_sum = sum(a)) but then ignoring it and recomputing sum(a) inside the loop anyway.
- Switching to a more mathematical formulation but accidentally reintroducing O(N^2) work via nested aggregates (e.g., sum( ) inside an O(N) loop).
- Introducing a global sort where the original task is solvable in O(N); while sometimes acceptable, it can be an unnecessary asymptotic regression for large N.
- Overusing greedy incremental strategies (select next best element via full scan) instead of leveraging structural simplifications like sign-enumeration or linearization.
- Applying convexity/unimodality assumptions (for early break) when the objective is not actually convex in p, which can lead to incorrect pruning.
- Restricting the parameter search range using constraints (e.g., [1, 100]) when such constraints are not guaranteed, changing the problem semantics.
- Using heavy Python data structures (tuples/lists of multiple ints per element) and repeated list allocations in inner loops, which keeps asymptotics the same but causes timeouts due to large constants.

## When not to use
- When input sizes are tiny (e.g., N <= 100 or small fixed), and the clarity of a straightforward double loop is more valuable than the complexity of math identities or sign-enumeration.
- When the scalar objective in p is not convex or unimodal and no safe structural property is known; using early termination based on assumed unimodality can miss the true optimum.
- When value-range constraints (e.g., all values in [1, 100]) are not guaranteed by the problem; hardcoding such ranges may silently change the problem and yield wrong results.
- When pairwise interactions include complex, non-linear dependencies that do not admit simple closed-form aggregation or prefix-sum formulations.
- When strict numeric stability or exactness requirements conflict with the chosen optimization (for example, using floating-point formulas where exact integer arithmetic was originally required).

## Minimal example
Before:
```py
def best_location(a):
    best_cost = float('inf')
    for p in range(min(a), max(a) + 1):  # redundant full range search
        cost = 0
        for x in a:                      # O(R * N)
            cost += abs(x - p)
        best_cost = min(best_cost, cost)
    return best_cost
```
After:
```py
def best_location(a):
    a_sorted = sorted(a)                 # median minimizes sum |x - p|
    m = a_sorted[len(a)//2]
    cost = sum(abs(x - m) for x in a)    # single O(N) aggregation
    return cost
```
