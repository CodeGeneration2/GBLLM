---
skill_id: O009
type: operator
language: python
family: state_compression
name: Arithmetic Loop to Closed Form Optimization
description: Turn heavy numeric loops into O(1) or O(√N) formulas by recognizing arithmetic progressions, floor-division patterns,
  and algebraic structure. This operator replaces brute-force iteration (including nested loops) with direct arithmetic, divisor/quotient
  grouping, and early-pruned enumeration, reducing both asymptotic complexity and constant factors while keeping space O(1)
  or near-input-size.
tags:
- math
- number_theory
- loop_optimization
- sqrt_decomposition
- arithmetic_progression
- divisor_sums
- closed_form
- constant_factor_optimization
- floor_division
- bruteforce_reduction
triggers:
- A while- or for-loop that repeatedly adds a constant step to a variable until crossing a numeric bound, and the result is
  the number of iterations or a simple function of that count.
- A loop `for i in range(1, n+1)` (or up to n//2) whose body uses integer division like `n // i` or slowly changing quotients/factors.
- A nested loop examining all pairs or triples of integers (or indices) with purely arithmetic conditions, especially when
  constraints allow values up to 1e7–1e12.
- A DP table or grid being filled with range sums by explicit loops over subranges, where each entry recomputes overlapping
  sums (suggesting prefix sums or 2D prefix sums).
- Triple nested loops over small fixed bounds (e.g., 100) evaluating a symmetric quadratic or combinatorial expression, often
  with conditions monotone in the innermost index.
- Loops whose inner body is only integer arithmetic and comparisons, with no dependence on complex state across iterations.
- A built-in aggregation (sum, max, min) or expensive operation inside a loop where the argument or part of the expression
  is invariant across iterations.
- A search over integer parameters to minimize or maximize a simple scalar objective (like count, distance, or affected cells)
  where the loop only uses the indices and constants, with no stateful updates.
---

## When to use
- A while- or for-loop that repeatedly adds a constant step to a variable until crossing a numeric bound, and the result is the number of iterations or a simple function of that count.
- A loop `for i in range(1, n+1)` (or up to n//2) whose body uses integer division like `n // i` or slowly changing quotients/factors.
- A nested loop examining all pairs or triples of integers (or indices) with purely arithmetic conditions, especially when constraints allow values up to 1e7–1e12.
- A DP table or grid being filled with range sums by explicit loops over subranges, where each entry recomputes overlapping sums (suggesting prefix sums or 2D prefix sums).
- Triple nested loops over small fixed bounds (e.g., 100) evaluating a symmetric quadratic or combinatorial expression, often with conditions monotone in the innermost index.
- Loops whose inner body is only integer arithmetic and comparisons, with no dependence on complex state across iterations.
- A built-in aggregation (sum, max, min) or expensive operation inside a loop where the argument or part of the expression is invariant across iterations.
- A search over integer parameters to minimize or maximize a simple scalar objective (like count, distance, or affected cells) where the loop only uses the indices and constants, with no stateful updates.

## Steps
1. Classify the loop pattern: If the loop increments a variable by a constant step until a bound, treat it as an arithmetic progression.
2. Classify the loop pattern: If the loop uses floor(n / i), treat it as a floor-division or divisor-sum pattern.
3. Classify the loop pattern: If it enumerates pairs/triples with symmetric formulas (e.g., quadratic in variables), treat it as a combinatorial or number-theoretic sum.
4. Classify the loop pattern: If it recomputes range sums in a grid, treat it as a prefix-sum candidate.
5. Extract the underlying math: For arithmetic progression loops, express the updated variable as `start + k * step` and the termination as an inequality in k.
6. Extract the underlying math: For floor-division loops, write the target sum explicitly as `∑ f(i, floor(n / i))` or in terms of divisor pairs (i, n//i).
7. Extract the underlying math: For pairwise sums/products, rewrite the brute-force double sum via algebraic identities (e.g., using total sum and sum of squares).
8. Solve or group analytically: Arithmetic progression

## Complexity
- Time: Typical transformations: O(K) loops over arithmetic progressions → O(1); O(N) floor-division or divisor-sum loops → O(√N); O(N^2) pairwise enumeration → O(N); O(N^3) grid/interval preprocessing → O(N^2) via prefix sums; plus significant
- Space: Usually maintained at O(1) extra space for closed-form and sqrt-decomposition methods, or O(N) / O(N^2) when replacing redundant DP/enumeration with 1D/2D prefix sums. Space rarely increases beyond the order already implied by the input

## Pitfalls
- Off-by-one errors when converting a loop `while current + step <= limit` or `for i in range(start, end+1)` into integer-division formulas; misplacing +1 or -1 terms around floor/ceil.
- Mismatching inclusive and exclusive bounds when computing counts; e.g., confusing the number of k values satisfying an inequality with the maximum k.
- Incorrect handling of negative values or sign when using integer division; many derivations assume non-negative ranges.
- Double-counting or missing terms when using divisor-pair symmetry or sqrt decomposition, especially on the diagonal where i * i = N or i = floor(N / i).
- For floor-division sums, failing to correctly group ranges of i where floor(N / i) is constant, leading to gaps or overlaps in coverage.
- Algebraic simplifications that rely on assumptions (like sorted data, fixed dimensions, or specific constraints) that do not hold for all valid inputs.
- Over-aggressive early breaks in nested loops without proving monotonicity in the break condition, which can skip valid solutions.
- Using floating-point division or math.floor for integer logic, which can introduce precision issues on large integers and unnecessary overhead.

## When not to use
- When loop iterations depend on complex mutable state, side effects, or non-monotone conditions (e.g., dynamic programming with intricate transitions, graph traversals), where there is no clear closed-form mapping.
- When input sizes are guaranteed to be very small, and the clarity of a straightforward loop is more valuable than a mathematically dense optimization.
- When arithmetic expressions involve arbitrary real numbers or non-linear functions without clear integer-progress or divisor structure; closed-form integer division tricks may not apply.
- When the code relies on probabilistic or approximate behavior where exact arithmetic formulas would change semantics.
- When memory budgets are extremely tight and introducing prefix arrays or grids would exceed limits; in such cases, prefer in-place or streaming methods over prefix-sum-based optimizations.
- When integer overflow is a concern in languages with fixed-width integers and the derived closed forms require large intermediate products (in Python this is less problematic, but in other languages it matters).

## Minimal example
Before:
```py
def sum_weighted_quotients(n: int) -> int:
    total = 0
    for i in range(1, n + 1):  # O(n) loop
        total += i * (n // i)
    return total
```
After:
```py
def sum_weighted_quotients(n: int) -> int:
    total = 0
    i = 1
    while i <= n:  # group ranges with same q = n // i, only O(sqrt(n)) steps
        q = n // i
        j = n // q
        total += q * (i + j) * (j - i + 1) // 2  # sum_{k=i}^j k * q
        i = j + 1
    return total
```
