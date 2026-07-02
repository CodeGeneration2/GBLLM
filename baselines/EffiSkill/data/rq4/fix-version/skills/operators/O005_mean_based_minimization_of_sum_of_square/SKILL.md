---
skill_id: O005
type: operator
language: python
family: quadratic_mean
name: Mean Based Minimization of Sum of Squared Deviations
description: 'Optimize brute-force searches over an integer parameter that minimize a sum of squared deviations by exploiting
  the quadratic structure: the cost is minimized at the mean of the data. Replace loops over all candidate integers in a numeric
  range with evaluating the objective at only floor(mean) and ceil(mean), using precomputed aggregates (sum and optionally
  sum of squares) to achieve linear time independent of the'
tags:
- optimization
- mathematical_optimization
- quadratic_minimization
- sum_of_squares
- least_squares
- mean
- variance_minimization
- array_processing
- search_space_reduction
- algorithmic_optimization
triggers:
- A loop over all integers between min(array) and max(array) (or 0..max(array)) where each iteration computes sum((a[i] -
  x)**2) or an equivalent quadratic cost for that x.
- Nested loops where the outer loop scans a value range derived from max(array) - min(array) and the inner loop scans all
  elements to recompute a sum of squared differences.
- Objective function has the form sum_i (a_i - p)^2 or can be algebraically expanded into n*p*p - 2*p*sum(a_i) + sum(a_i^2)
  for some scalar parameter p.
- Time complexity depends on the numeric spread of values (max - min or max value) rather than on the number of elements n.
- Use of NumPy or other vectorized operations inside a Python loop over candidate parameter values to repeatedly compute full-array
  squared differences.
- An unused or redundant outer loop that repeats an identical search over candidate centers without using its loop variable.
- Code initializes a best_cost to +inf and then updates it by brute-force over many integer candidates of a scalar parameter.
- Sorting solely to obtain min and max before scanning all integers in that range and recomputing squared deviation sums for
  each.
---

## When to use
- A loop over all integers between min(array) and max(array) (or 0..max(array)) where each iteration computes sum((a[i] - x)**2) or an equivalent quadratic cost for that x.
- Nested loops where the outer loop scans a value range derived from max(array) - min(array) and the inner loop scans all elements to recompute a sum of squared differences.
- Objective function has the form sum_i (a_i - p)^2 or can be algebraically expanded into n*p*p - 2*p*sum(a_i) + sum(a_i^2) for some scalar parameter p.
- Time complexity depends on the numeric spread of values (max - min or max value) rather than on the number of elements n.
- Use of NumPy or other vectorized operations inside a Python loop over candidate parameter values to repeatedly compute full-array squared differences.
- An unused or redundant outer loop that repeats an identical search over candidate centers without using its loop variable.
- Code initializes a best_cost to +inf and then updates it by brute-force over many integer candidates of a scalar parameter.
- Sorting solely to obtain min and max before scanning all integers in that range and recomputing squared deviation sums for each.

## Steps
1. Recognize quadratic structure: Identify that the objective is to minimize a sum of squared deviations of the form S(p) = sum_i (a_i - p)**2 with respect to a single scalar parameter p (often integer-constrained). If necessary, expand the square: S(p) = sum_i
2. Derive the continuous minimizer: Use calculus or algebra on S(p) = n*p**2 - 2*p*sum_a + sum_a2. The derivative is dS/dp = 2*n*p - 2*sum_a, which is zero at p* = sum_a / n, i.e., the arithmetic mean of the data.
3. Precompute aggregates in one pass: Traverse the array once to compute at least S = sum(a_i) and optionally S2 = sum(a_i**2). Keep these in integer type to avoid precision issues for large values. This pass is O(n).
4. Evaluate objective at candidate integers: Either: a) Direct loop form (no algebraic expansion): for each candidate p in {p1, p2}, compute cost_p = sum((x - p)**2 for x in a) in a single O(n) pass, or b) Closed-form form: use cost_p = n*p*p - 2*p*S + S2
5. Remove brute-force search: Eliminate any outer loop over the integer range [min(a), max(a)] or [0, max(a)] and replace it with just the constant-size candidate evaluation described above. Also remove redundant loops whose index is unused.
6. Avoid ineffective vectorization and repeated global stats: If using a numerical library, ensure you perform at most a constant number of full-array operations (e.g., one or two vectorized squared-difference sums), not one per candidate value. Precompute

## Complexity
- Time: Optimized: O(n) time (one pass to compute aggregates, plus one or two O(n) passes or O(1) evaluations for candidate centers). Unoptimized brute-force patterns in this cluster typically run in O(n * R) or worse, where R is the numeric
- Space: (pattern dependent)

## Pitfalls
- Retaining range-based loops: Only partially applying the optimization by computing the mean but still iterating over the full [min, max] or [0, max] range instead of restricting to floor/ceil of the mean.
- Incorrect handling of integer rounding: Using naive round() with banker's rounding when the required behavior is floor/ceil or ROUND_HALF_UP; this can shift the chosen integer by 1 and change the optimal cost in tie
- Ignoring the need to check both neighbors: Assuming that simply int(mean) or round(mean) is always optimal for integer p and failing to also evaluate the adjacent integer, which can be wrong when the mean is exactly
- Floating-point precision issues: Computing the mean in floating point for very large sums and then squaring large integers can introduce rounding errors. For robustness, prefer integer aggregates (sum, sum of squares)
- Unnecessary or misused vectorization: Calling vectorized operations (e.g., NumPy) inside a Python loop over candidate values leads to O(n * R) time and heavy allocation overhead, negating the analytical optimization.
- Recomputing global statistics: Repeatedly recomputing min(a), max(a), sum(a), or sum(a**2) inside loops; these should be computed once outside any candidate evaluation loop.
- Overlooking unused loops or variables: Leaving in an outer loop whose index is not used, causing an accidental extra factor of R in complexity (e.g., turning O(n * R) into O(n * R^2)).
- Using sorting unnecessarily: Sorting the array just to obtain min and max or to run a brute-force search on a range; after switching to a mean-based approach, sorting is typically unnecessary and wastes O(n log n) time.

## When not to use
- The parameter is multi-dimensional or coupled: When optimizing over vectors or multiple parameters jointly, or when the cost cannot be separated as a simple quadratic in a single scalar p, this one-dimensional
- Discrete objective is not convex in p: If the cost as a function of p is not convex (or not essentially quadratic), then evaluating only floor(mean) and ceil(mean) may fail; brute-force search or other optimization

## Minimal example
Before:
```py
positions = [3, 7, 11, 14, 18]
mn, mx = min(positions), max(positions)
best_x, best_cost = None, float('inf')
for x in range(mn, mx + 1):
    cost = sum((p - x) ** 2 for p in positions)
    if cost < best_cost:
        best_x, best_cost = x, cost
```
After:
```py
positions = [3, 7, 11, 14, 18]
n = len(positions)
S = sum(positions)
for x in (S // n, (S + n - 1) // n):
    cost = sum((p - x) ** 2 for p in positions)
    if 'best_x' not in locals() or cost < best_cost:
        best_x, best_cost = x, cost
```
