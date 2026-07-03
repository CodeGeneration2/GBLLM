---
skill_id: O027
type: operator
language: python
family: coprimality
name: Coprimality Classification Optimization
description: Optimize code that classifies integer arrays by coprimality (e.g., pairwise vs setwise) by replacing naïve pairwise
  gcd checks or per-number trial division with range-based sieves, smallest-prime-factor (SPF) tables, or frequency+divisor
  sweeps. The operator focuses on choosing the right global strategy (sieve vs on-demand factorization vs divisor-aggregation),
  adapting precomputation to the true maximum value
tags:
- number-theory
- gcd
- prime-factorization
- sieve
- frequency-counting
- divisor-iteration
- coprimality
- algorithmic-optimization
- constant-factor-optimization
- python-performance
triggers:
- Code checks pairwise coprimality using nested gcd or divisibility loops over all pairs of numbers.
- Each number is factorized independently via trial division up to sqrt(max_value) with many modulo operations.
- A full sieve or SPF table is built up to a fixed large constant bound (e.g., 10**6) regardless of the actual maximum input
  value.
- Large global arrays (size ~max_possible_value) are allocated even when the actual maximum input value is much smaller.
- Coprimality logic only needs to know whether some prime divides multiple numbers, but the implementation fully factorizes
  all numbers and counts all multiplicities.
- Presence of a loop over all potential divisors d in [2..U] combined with an inner multiples loop (or list slicing like arr[d::d])
  regardless of how many input values are actually multiples of d.
- Detection of a condition such as 'shared divisor found' that sets a flag but does not break out of heavy loops early.
- Per-number factorization creates fresh sets or dictionaries for prime factors and updates global defaultdicts or sets in
  inner loops.
---

## When to use
- Code checks pairwise coprimality using nested gcd or divisibility loops over all pairs of numbers.
- Each number is factorized independently via trial division up to sqrt(max_value) with many modulo operations.
- A full sieve or SPF table is built up to a fixed large constant bound (e.g., 10**6) regardless of the actual maximum input value.
- Large global arrays (size ~max_possible_value) are allocated even when the actual maximum input value is much smaller.
- Coprimality logic only needs to know whether some prime divides multiple numbers, but the implementation fully factorizes all numbers and counts all multiplicities.
- Presence of a loop over all potential divisors d in [2..U] combined with an inner multiples loop (or list slicing like arr[d::d]) regardless of how many input values are actually multiples of d.
- Detection of a condition such as 'shared divisor found' that sets a flag but does not break out of heavy loops early.
- Per-number factorization creates fresh sets or dictionaries for prime factors and updates global defaultdicts or sets in inner loops.

## Steps
1. Determine constraints and choose a global strategy: If max_value (A_max) is moderately bounded (e.g., ≤ 1e6) and n is large, prefer a sieve/SPF or frequency+divisor aggregation approach.
2. Determine constraints and choose a global strategy: If n is large but A_max is fixed and relatively small, consider global frequency + divisor sweeps or global prime counting.
3. Determine constraints and choose a global strategy: If n is moderate and A_max is bounded, consider an SPF sieve or a small prime table for per-number factorization.
4. Adapt precomputation range to the data: Compute A_max = max(array).
5. Adapt precomputation range to the data: Allocate any sieve/SPF/frequency arrays to size A_max+1 instead of a hard-coded bound.
6. Adapt precomputation range to the data: Avoid multiple independent arrays over the full range; reuse or consolidate where possible.
7. If using SPF-based factorization: Build a smallest-prime-factor (SPF) array spf[0..A_max] using a standard sieve (for p in 2..A_max, if spf[p]==0 then mark multiples).
8. If using SPF-based factorization: Ensure each composite is assigned its SPF only once to minimize redundant writes.

## Complexity
- Time: (pattern dependent)
- Space: O(A_max) for sieve/SPF and frequency-array approaches (single or few arrays sized to max(input)), or O(π(√A_max)) ≈ O(1) when using a fixed small prime table plus per-number factor lists or sets. Global state is kept proportional to the

## Pitfalls
- Using O(n^2) pairwise gcd/divisibility checks instead of aggregating information globally.
- Applying trial division up to sqrt(A_max) for every number when n and A_max are both large, leading to ~1e8 modulo operations in Python.
- Building a full-range sieve or SPF table up to a hard-coded bound (e.g., 10**6) even when max(input) is much smaller, wasting time and memory.
- Precomputing multiple large arrays (SPF, is_prime, frequency, last_index) over the entire range instead of a single compact structure.
- Fully factorizing every number and counting all prime multiplicities when classification only requires detecting whether a prime appears in two or more numbers.
- Marking all multiples of a prime factor up to A_max for each number, even though only the prime itself needs to be tracked for coprimality.
- Using Python sets or defaultdict in inner loops for prime-factor tracking instead of array-based counters or boolean markers.
- Relying on list slicing (arr[d::d]) inside heavy loops, causing repeated allocations and increasing constant factors.

## When not to use
- When the maximum value A_max is extremely large relative to n (e.g., 10**12 or higher) so that array-based sieves or frequency tables of size A_max are impossible; in such cases rely on pure per-number factorization
- When the language or environment supports highly optimized low-level loops (e.g., C++ with built-in sieves) and Python-specific overhead is not a concern; then simpler sieve-based patterns may already be sufficient.
- When the task does not require global coprimality classification (pairwise vs setwise) but only local gcds or simple divisibility checks; bringing in global sieve/frequency machinery may overcomplicate the solution.
- When n is tiny (e.g., n <= 100) and constraints are loose; straightforward pairwise gcd or naïve trial division may be simpler and completely adequate.
- When memory is extremely constrained and even O(A_max) arrays for A_max around 10**6 are too large; prefer constant-memory prime-table trial division or streaming gcd-only approaches.

## Minimal example
Before:
```py
from math import gcd

def classify(nums):
    n = len(nums)
    max_g = 0
    for i in range(n):
        for j in range(i + 1, n):
            max_g = max(max_g, gcd(nums[i], nums[j]))
    overall_g = nums[0]
    for x in nums[1:]:
        overall_g = gcd(overall_g, x)
    if max_g == 1 and overall_g == 1:
        return "pairwise"
    if overall_g == 1:
        return "setwise"
    return "not"
```
After:
```py
from math import gcd

def classify(nums):
    overall_g = 0
    for x in nums:
        overall_g = gcd(overall_g, x)
    if overall_g != 1:
        return "not"
    A_max = max(nums)
    freq = [0] * (A_max + 1)
    for x in nums:
        freq[x] += 1
    spf = list(range(A_max + 1))
    for i in range(2, int(A_max ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, A_max + 1, i):
                if spf[j] == j:
                    spf[j] = i
    used = set()
    for x in nums:
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            x //= p
        if any(p in used for p in factors):
            return "setwise"
        used.update(factors)
    return "pairwise"
```
