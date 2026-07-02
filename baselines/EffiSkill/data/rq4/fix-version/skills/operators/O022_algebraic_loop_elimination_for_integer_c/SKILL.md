---
skill_id: O022
type: operator
language: python
family: combinatorics
name: Algebraic Loop Elimination for Integer Combinatorics
description: 'Transform brute-force integer enumeration with nested loops into faster code by exploiting linear relations
  between loop variables. When counting or checking integer tuples (pairs, triples, etc.) under constraints like a + b + c
  = S or linear inequalities, treat some variables as dependent: compute them algebraically from the others instead of looping
  over them. Combine this with tighter loop bounds, early'
tags:
- loop-optimization
- algorithmic-optimization
- combinatorics
- integer-arithmetic
- nested-loops
- sum-constraints
- search-space-reduction
- closed-form
- two-pointer-technique
- vectorization
triggers:
- Triple or double nested loops over integer ranges with bounds tied to the same size parameter, where the innermost body
  only does simple arithmetic and comparisons.
- A linear equality or inequality linking loop variables (e.g., a + b + c = S, or X >= k*(p + q) + q) checked inside the innermost
  loop.
- Tasks that only need counts or existence (how many tuples satisfy the constraints) rather than listing all tuples.
- Explicit enumeration of all ordered or strictly ordered integer tuples (such as i < j < k) under a fixed-sum constraint.
- Innermost loops that linearly search for a value that is mathematically determined by the outer-loop variables and a constant.
- Use of sum([ ]) or similar function calls over tiny fixed-size lists inside hot loops, especially with three nested loops.
- Use of data structures like list(range(1, n+1)) solely to iterate sequential integers that can be produced directly by loop
  indices.
- Loops over a fixed numeric bound (e.g., up to 1e5) that check a monotone linear inequality and break on the first satisfying
  value.
---

## When to use
- Triple or double nested loops over integer ranges with bounds tied to the same size parameter, where the innermost body only does simple arithmetic and comparisons.
- A linear equality or inequality linking loop variables (e.g., a + b + c = S, or X >= k*(p + q) + q) checked inside the innermost loop.
- Tasks that only need counts or existence (how many tuples satisfy the constraints) rather than listing all tuples.
- Explicit enumeration of all ordered or strictly ordered integer tuples (such as i < j < k) under a fixed-sum constraint.
- Innermost loops that linearly search for a value that is mathematically determined by the outer-loop variables and a constant.
- Use of sum([ ]) or similar function calls over tiny fixed-size lists inside hot loops, especially with three nested loops.
- Use of data structures like list(range(1, n+1)) solely to iterate sequential integers that can be produced directly by loop indices.
- Loops over a fixed numeric bound (e.g., up to 1e5) that check a monotone linear inequality and break on the first satisfying value.

## Steps
1. Identify algebraic relationships: Inspect conditions inside the innermost loop for linear equations or inequalities (such as a + b + c == S, a + b + c <= S, or X >= k*(p + q) + q).
2. Identify algebraic relationships: Confirm that once some variables are fixed, the others are uniquely determined or belong to a short interval determined by simple arithmetic.
3. Choose independent loop variables and eliminate dependents: a
4. Choose independent loop variables and eliminate dependents: b).
5. Choose independent loop variables and eliminate dependents: i
6. Choose independent loop variables and eliminate dependents: j) and simple bound/order checks (e.g., j < k <= n).
7. Tighten loop bounds using constraints: .

## Complexity
- Time: Typically reduces cubic or higher enumeration (e.g., O(n^3) over triplets) to quadratic or linear time (e.g., O(n^2) over pairs with algebraically computed third element, or O(n) / O(1) via closed-form formulas). For some structured
- Space: Generally keeps space O(1) by counting on the fly instead of storing tuples. In some refactorings that use list comprehensions solely to count, space may temporarily rise to O(n^2) or O(#solutions), but this is avoidable by using counters

## Pitfalls
- Breaking correctness of ordering or bounds when eliminating loops: a
- Breaking correctness of ordering or bounds when eliminating loops: b), forgetting to enforce original constraints like a < b < c, positivity, or c <= n.
- Breaking correctness of ordering or bounds when eliminating loops: Misaligned loop ranges that silently skip valid tuples or double-count because inequalities are not equivalent to the original nested-loop structure.
- Off-by-one errors in bound derivations: lower + 1).
- Off-by-one errors in bound derivations: Using floor/ceil incorrectly when turning continuous inequalities into integer ranges.
- Overlooking overflow or integer division nuances: q)/(p + q))).
- Overlooking overflow or integer division nuances: Failing to handle negative or zero denominators when deriving formulas from inequalities.
- Introducing unnecessary materialization and memory usage: Replacing a triple loop with a double loop but still building a list of size O(n^2) just to take its length, instead of using a counter.

## When not to use
- When all tuples must be explicitly listed, not just counted: Algebraic elimination of variables is less beneficial if you must materialize every valid combination anyway; reducing loop depth might still help, but you
- When constraints are highly non-linear or non-algebraic: If conditions involve complex non-linear relationships, arbitrary predicates, or black-box checks, there may be no simple way to express one variable in terms of
- When asymptotic complexity is inherently high and cannot be reduced by algebra: For problems where the true complexity is exponential or combinatorial without exploitable linear structure, algebraic loop elimination
- When vectorized or library-based approaches do not fit the environment: If external numerical libraries are unavailable or disallowed, or if memory limits are tight, you may not be able to apply vectorization or large

## Minimal example
Before:
```py
def count_triangles(p: int) -> int:
    count = 0
    for a in range(1, p):
        for b in range(1, p):
            for c in range(1, p):
                if a <= b <= c and a + b > c and a + b + c == p:
                    count += 1
    return count
```
After:
```py
def count_triangles(p: int) -> int:
    count = 0
    for a in range(1, p // 3 + 1):
        for b in range(a, (p - a) // 2 + 1):
            c = p - a - b  # algebraic elimination: a + b + c == p
            if b <= c and a + b > c:
                count += 1
    return count
```
