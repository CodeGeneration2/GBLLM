---
skill_id: O016
type: operator
language: python
family: combinatorics
name: Eliminate Bruteforce Triples via Algebraic Counting
description: Replace cubic-time triple nested loops used for integer/combinatorial counting with lower-dimensional iteration
  or closed-form arithmetic by exploiting linear constraints (such as fixed sums) and ordering. Express one variable algebraically
  in terms of others, tighten loop bounds from inequalities, and, when possible, count solutions directly instead of enumerating
  them.
tags:
- algorithmic_optimization
- time_complexity_reduction
- combinatorics
- counting
- loop_optimization
- triple_loops
- sum_constraint
- math_reformulation
- two_pointer_style
- closed_form_counting
triggers:
- Three nested loops over small or moderate integer ranges (e.g., 1..n) performing only simple arithmetic and comparisons
  in the innermost body.
- A linear equality constraint tying loop indices together, such as i + j + k = S or a1*x + a2*y + a3*z = S.
- Strict ordering or distinctness constraints between indices (e.g., i < j < k) where only counts, not actual tuples, are
  needed.
- The innermost loop index appears only in the sum/equality condition and simple range checks, indicating it can be solved
  for algebraically.
- Use of sum([ ]) or repeated allocation (lists, temporary arrays) inside the innermost loop in a pure counting task.
- Observed or anticipated performance issues consistent with O(n^3) growth (e.g., n up to 10^2–10^3, many test cases, timeouts).
- Cubic search over a fixed small value range to satisfy linear or affine relationships on a tiny grid or matrix.
- Brute-force enumeration of all solutions followed by using only their count (e.g., appending to a list and later taking
  its length).
---

## When to use
- Three nested loops over small or moderate integer ranges (e.g., 1..n) performing only simple arithmetic and comparisons in the innermost body.
- A linear equality constraint tying loop indices together, such as i + j + k = S or a1*x + a2*y + a3*z = S.
- Strict ordering or distinctness constraints between indices (e.g., i < j < k) where only counts, not actual tuples, are needed.
- The innermost loop index appears only in the sum/equality condition and simple range checks, indicating it can be solved for algebraically.
- Use of sum([ ]) or repeated allocation (lists, temporary arrays) inside the innermost loop in a pure counting task.
- Observed or anticipated performance issues consistent with O(n^3) growth (e.g., n up to 10^2–10^3, many test cases, timeouts).
- Cubic search over a fixed small value range to satisfy linear or affine relationships on a tiny grid or matrix.
- Brute-force enumeration of all solutions followed by using only their count (e.g., appending to a list and later taking its length).

## Steps
1. Identify structure and goal: Confirm the task is counting (or existence checking) of integer triples or small-dimensional tuples under linear constraints.
2. Identify structure and goal: Verify that the third (or last) variable only appears in linear equations and range/order checks, and that tuples need not be explicitly listed.
3. Derive a dependent variable: i
4. Derive a dependent variable: j).
5. Derive a dependent variable: a1*x
6. Derive a dependent variable: a2*y) / a3) and encode divisibility via indexing or stride logic instead of a modulus in a loop.
7. Reduce loop dimensionality: Replace the triple nested loop over (i, j, k) with a double loop over two variables (e.g., (i, j)), computing the third via the derived formula.
8. Reduce loop dimensionality: Alternatively, fix one variable (often the largest or the middle element) and use a single loop over it while counting valid pairs of the remaining variables via a constant-time helper.

## Complexity
- Time: Typically improves from O(n^3) to O(n^2) or O(n) per test case for triple-sum-style counting; in fixed-size or heavily constrained domains, can reach O(1). For array-based/vectorized reforms, arithmetic work may remain O(S^2) but
- Space: Usually O(1) extra space beyond input; O(n) or O(S) when using DP or frequency arrays; up to O(n^2) if materializing all pairs or using dense 2D structures. Most algebraic counting variants remain constant-space.

## Pitfalls
- Incorrect inequality translation: Miscomputing bounds (off-by-one errors) when deriving intervals for remaining variables (e.g., j_min/j_max or i_min/i_max).
- Incorrect inequality translation: Forgetting to enforce strict inequalities like i < j < k after eliminating one loop, leading to duplicate or invalid tuples.
- Range and feasibility oversights: Failing to ensure the derived variable is within its domain (e.g., k must be an integer, positive, and <= n).
- Range and feasibility oversights: Not shrinking outer loop ranges using obvious constraints (e.g., looping k from 1..n instead of starting near S/3). This preserves correctness but undercuts performance gains.
- Algebraic mistakes: Incorrect rearrangement of the sum or polynomial, leading to miscounted or missed solutions.
- Algebraic mistakes: Assuming monotonicity or linearity where it does not hold, causing invalid pruning or early breaks.
- Overcomplication vs. constraints: Overengineering closed-form formulas when n is extremely small and the cubic solution already fits comfortably within time limits.
- Overcomplication vs. constraints: Introducing complex arithmetic that is harder to maintain and more error-prone than a simpler quadratic solution that already meets constraints.

## When not to use
- When all solutions must be explicitly listed rather than counted, and downstream logic depends on each tuple individually.
- When constraints are non-linear, non-monotone, or involve complex predicates that cannot be captured via simple algebraic relations or inequalities.
- When n and the search ranges are extremely small (e.g., n <= 20 and few test cases), where cubic enumeration is simpler and comfortably fast.
- When adding algebraic counting significantly increases code complexity without providing needed performance benefits for the given input limits.
- When constraints involve arbitrary data-dependent conditions (e.g., looking up values in large arrays or graphs) that prevent solving one variable directly from others.

## Minimal example
Before:
```py
# O016 focus: eliminate
cnt = 0
for a in range(1, n+1):
    for b in range(1, n+1):
        for c in range(1, n+1):
            if a + b + c == S: cnt += 1
```
After:
```py
# optimized for eliminate
cnt = 0
for a in range(1, n+1):
    lo = max(1, S-a-n); hi = min(n, S-a-1)
    if lo <= hi: cnt += (hi - lo + 1)
```
