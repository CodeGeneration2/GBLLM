---
skill_id: O024
type: operator
language: python
family: digit
name: Digit Range Closed Form Counting
description: Replace per-integer digit-processing loops with arithmetic over digit-length ranges. When a count or aggregate
  depends only on properties that are constant within ranges of equal digit length (or similar magnitude buckets), derive
  a piecewise closed-form formula in terms of range boundaries instead of iterating from 1 to N.
tags:
- optimization
- closed-form
- digit-processing
- range-counting
- math-derivation
- loop-elimination
- constant-time
- counting
- branching
- python-performance
triggers:
- A loop iterates from 1 to N and, for each integer, computes the number of digits via string conversion, logarithms, or repeated
  division.
- The property being tested depends only on digit length or simple magnitude ranges (e.g., odd/even digit count) rather than
  on the specific digits.
- Constraints allow N to be large (e.g., up to 1e5 or more), making O(N log N) or O(N) with heavy per-iteration work risky
  in Python.
- Digit-based thresholds such as 9, 99, 999, 9999, naturally partition the domain and the result is just a count over these
  partitions.
- There is unnecessary precomputation over a fixed large bound (e.g., up to 10^5) building lists or dicts of numbers classified
  by digit properties.
- A helper function is called inside the main loop to count digits via repeated division by 10.
- The final answer is a single scalar (count or simple aggregate) and does not require constructing or returning the full
  list of qualifying numbers.
---

## When to use
- A loop iterates from 1 to N and, for each integer, computes the number of digits via string conversion, logarithms, or repeated division.
- The property being tested depends only on digit length or simple magnitude ranges (e.g., odd/even digit count) rather than on the specific digits.
- Constraints allow N to be large (e.g., up to 1e5 or more), making O(N log N) or O(N) with heavy per-iteration work risky in Python.
- Digit-based thresholds such as 9, 99, 999, 9999, naturally partition the domain and the result is just a count over these partitions.
- There is unnecessary precomputation over a fixed large bound (e.g., up to 10^5) building lists or dicts of numbers classified by digit properties.
- A helper function is called inside the main loop to count digits via repeated division by 10.
- The final answer is a single scalar (count or simple aggregate) and does not require constructing or returning the full list of qualifying numbers.

## Steps
1. Identify the exact property or count being computed and confirm it depends only on digit length or magnitude buckets (e.g., depends on number of digits, not on detailed digit patterns).
2. Partition the integer domain into contiguous digit-length ranges: [1, 9], [10, 99], [100, 999], [1000, 9999], or, more generally, [10^(k-1), 10^k - 1] for k-digit numbers.
3. Determine which ranges contribute non-zero to the answer (e.g., only odd-digit lengths, or only up to some maximum length implied by constraints).
4. For each contributing range, derive a formula for its contribution to the answer as a function of N: typically an intersection size like max(0, min(N, upper) - lower + 1) multiplied by any constant per-element weight if needed.
5. Algebraically simplify the sum of range contributions into a small set of piecewise expressions based on comparisons of N with powers of 10 (or analogous boundaries).
6. Encode the simplified logic as a short if/elif chain (or equivalent branching) with only integer comparisons, additions, subtractions, and min/max; avoid any per-number loops or string/logarithm calls.
7. Remove all unnecessary precomputation structures (lists, dicts, sets) and per-element digit computations; keep only scalar variables updated in constant time.
8. Optionally generalize to arbitrary N by generating digit-range boundaries on the fly (e.g., loop over k up to number of digits of N) while keeping time O(log N) instead of O(N).

## Complexity
- Time: Typically O(1) for fixed maximum digit length (using a finite set of hard-coded ranges); more generally O(log N) if ranges are generated from powers of 10 on the fly.
- Space: O(1) additional space; only a few scalar variables and constants are needed.

## Pitfalls
- Deriving incorrect range bounds, especially off-by-one errors at powers of 10 (e.g., mixing up 99 vs 100 or using N - 100 instead of N - 99).
- Forgetting to clamp partial ranges with min/max so that when N lies inside a range, the contribution does not exceed the range size or become negative.
- Hard-coding range cases only for a specific constraint (e.g., up to 10^5) and silently breaking when used with larger N in a different context.
- Assuming the property is constant over digit-length ranges when it actually depends on finer digit patterns (e.g., specific digits, sums of digits, or forbidden digits), where a simple closed-form over full ranges does
- Mixing inclusive and exclusive bounds inconsistently, leading to off-by-one errors in the final count.
- Over-optimizing to a purely constant-time formula when the problem requires handling arbitrary large N, where an O(log N) range-iteration based on powers of 10 would be simpler and safer.
- Removing loops but retaining unnecessary string conversions, logarithms, or repeated function calls, which leaves significant constant-factor overhead.
- Ignoring that some alternative formulations (e.g., combinatorial enumeration instead of digit DP) can blow up for larger parameters, even if they are fine for very small fixed bounds.

## When not to use
- When the property depends on detailed digit patterns that vary within a digit-length range (e.g., specific digits, digit sums, forbidden patterns), and no simple closed-form aggregation over full ranges is available.
- When N is small and simplicity or clarity is more important than micro-optimizations; a straightforward loop may be more maintainable.
- When the parameter space allows large combinatorial explosion for closed-form enumeration (e.g., enumerating all assignments of non-zero digits), and a DP-based approach has better asymptotic behavior for larger
- When the constraints or domain may change (e.g., N can become much larger) and the optimization relies on hard-coded bounds or ranges that would no longer be valid.
- When you need to output or inspect each qualifying number, not just count them; closed-form counting alone is insufficient in that case.

## Minimal example
Before:
```py
def count_odd_digit_numbers(n: int) -> int:
    count = 0
    for x in range(1, n + 1):
        if len(str(x)) % 2 == 1:  # depends only on digit length
            count += 1
    return count
```
After:
```py
def count_odd_digit_numbers(n: int) -> int:
    count = 0
    length = 1
    while 10 ** (length - 1) <= n:
        low, high = 10 ** (length - 1), min(n, 10 ** length - 1)
        if length % 2 == 1:
            count += max(0, high - low + 1)
        length += 1
    return count
```
