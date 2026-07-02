---
skill_id: O011
type: operator
language: python
family: digit
name: Optimize Per Digit Loops in Brute Force Number Ranges
description: When iterating over large numeric ranges and doing per-digit work (digit count, digit sum, digit-based predicates),
  naive implementations often use inner while-loops with modulo/division, float arithmetic, helper-function calls per element,
  or unnecessary precomputation structures. This operator rewrites such hot paths to use safe integer-only arithmetic or C-implemented
  string operations, flattens helper calls
tags:
- optimization
- loops
- digit-processing
- integer-arithmetic
- string-builtins
- constant-factor
- bugfix
- bruteforce
- counting
- implementation
triggers:
- A tight outer loop over a numeric range (e.g., for x in range(1, N+1)) with an inner while-loop that strips digits via `%
  10` and `/ 10` or `//= 10`.
- Use of `/` or `/=` on a variable that conceptually represents an integer being reduced digit-by-digit, combined with a termination
  condition like `while x != 0` or `while x > 0`.
- Digit operations implemented via float arithmetic plus `math.floor` (e.g., `math.floor(x % 10)` or `x /= 10` inside the
  inner loop).
- A small helper function that performs per-digit arithmetic and is called once for each element of a large range, dominating
  runtime.
- Repeated use of `len(str(x))`, `list(str(x))`, or similar conversions inside a large loop, or separate precomputation over
  a fixed bound (e.g., 1..100000) to cache simple digit properties.
- Code that appears to hang or take excessively long for small inputs due to a non-terminating or extremely long inner digit
  loop driven by float underflow.
- Large temporary lists or dictionaries built only to encode simple digit properties (like odd digit count) that could be
  computed on-the-fly.
---

## When to use
- A tight outer loop over a numeric range (e.g., for x in range(1, N+1)) with an inner while-loop that strips digits via `% 10` and `/ 10` or `//= 10`.
- Use of `/` or `/=` on a variable that conceptually represents an integer being reduced digit-by-digit, combined with a termination condition like `while x != 0` or `while x > 0`.
- Digit operations implemented via float arithmetic plus `math.floor` (e.g., `math.floor(x % 10)` or `x /= 10` inside the inner loop).
- A small helper function that performs per-digit arithmetic and is called once for each element of a large range, dominating runtime.
- Repeated use of `len(str(x))`, `list(str(x))`, or similar conversions inside a large loop, or separate precomputation over a fixed bound (e.g., 1..100000) to cache simple digit properties.
- Code that appears to hang or take excessively long for small inputs due to a non-terminating or extremely long inner digit loop driven by float underflow.
- Large temporary lists or dictionaries built only to encode simple digit properties (like odd digit count) that could be computed on-the-fly.

## Steps
1. Identify the hot path: locate the outer iteration over a numeric range and any inner loop or helper used to compute digit-based information (digit count, digit sum, last digit, etc.). Use profiling or inspection to confirm this dominates runtime.
2. Eliminate floating-point digit logic: if the inner loop uses `/` or `/=` on an integer, replace it with integer floor division `//= 10` and keep the loop variable as an int. Remove `math.floor` around `% 10` when working with non-negative integers, using
3. Choose the appropriate digit strategy: For correctness and predictable performance with minimal allocations, implement digit processing with pure integer arithmetic: `while y != 0: digit = y % 10; ; y //= 10`.
4. Choose the appropriate digit strategy: Digit count: `digits = len(str(x))`.
5. Choose the appropriate digit strategy: Digit sum: `s = sum(map(int, str(x)))`. Decide per use-case whether integer arithmetic or string-based processing is clearer and faster.
6. Inline trivial helpers: if a helper function is called for each element and only performs digit counting/summing, inline its logic directly into the main loop or replace it with a builtin expression (`len(str(x))`, `sum(map(int, str(x)))`) to remove Python
7. Remove redundant precomputation and data structures: delete fixed-range precomputation loops and auxiliary lists/dicts that store numbers or digit properties when the same property can be computed on-the-fly. Maintain only scalar counters and loop variables
8. Optionally vectorize with comprehensions and builtins: where memory permits, move per-element filters into comprehensions (e.g., `[i for i in range(1, N+1) if predicate(i)]`) and then aggregate (e.g., `len( )` or `sum( )`), leveraging C-level loops instead

## Complexity
- Time: Typically preserves the big-O (e.g., O(N * D) or O(N log N) where D is digit count) but reduces constant factors substantially by avoiding float arithmetic, math-module calls, unnecessary helper invocations, and redundant precomputation
- Space: Often reduces space from O(B) or O(N) (due to precomputed lists/dicts or materialized result lists) to O(1) extra space using scalar accumulators, though string-based methods introduce transient O(D) allocations per element. If using

## Pitfalls
- Using `/` or `/=` instead of `//` or `//=`, which silently promotes an integer loop variable to float, causing slow float division, non-terminating loops that rely on underflow to reach 0, and incorrect digit sums.
- Keeping `math.floor` or other `math.*` calls around digit extraction when all values are integers; this adds overhead with no benefit and often hides unintended float usage.
- Over-allocating temporary objects: repeatedly building `list(str(x))` or similar per iteration when only the length or sum of digits is needed, or creating large lists/dicts to cache simple digit properties for later
- Trading time for space blindly: rewriting loops as list comprehensions that materialize all candidates into a list for later counting or summing can increase memory usage from O(1) to O(N) and may be slower for large N.
- Over-optimizing digit arithmetic in Python when constraints are small: complex integer-based loops may be less readable and not significantly faster than straightforward string-based solutions for modest input sizes.
- Assuming asymptotic complexity alone determines performance: even when time complexity is formally unchanged (e.g., O(N log N)), float underflow loops or math-module calls can introduce huge constant factors that cause
- Failing to adjust loop bounds when refactoring: keeping `range(N+1)` or starting at 0 when 0 is never needed adds unnecessary iterations and can mask off-by-one or logic errors.

## When not to use
- When the algorithm is not bottlenecked by per-digit processing, and profiling shows other components (I/O, sorting, complex DP) dominate runtime.
- When input sizes are tiny and clarity is more important than micro-optimizations; a simple, readable string-based solution may suffice without eliminating helper functions or restructuring loops.
- When a more substantial algorithmic improvement is available (e.g., closed-form counting of numbers by digit properties) that avoids iterating over all numbers entirely; in such cases, focus on changing the overall
- When memory is very constrained and building intermediate lists for vectorized comprehensions would change space from O(1) to O(N) in a critical context; prefer in-place counting with scalar accumulators instead.
- When digit operations must be done in a non-decimal base or require arbitrary precision behavior beyond typical int/string conversions; more specialized arithmetic or libraries may be needed instead of generic

## Minimal example
Before:
```py
# O011 focus: optimize
ans = 0
for i in range(1, n+1):
    ans += len(str(i))
```
After:
```py
# optimized for optimize
ans = 0
for d in range(1, 19):
    L, R = 10**(d-1), min(n, 10**d - 1)
    if L <= R: ans += (R - L + 1) * d
```
