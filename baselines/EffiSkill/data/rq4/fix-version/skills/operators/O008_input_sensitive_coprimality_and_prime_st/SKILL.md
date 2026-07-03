---
skill_id: O008
type: operator
language: python
family: coprimality
name: Input sensitive coprimality and prime structure classification
description: A reusable optimization pattern for classifying integer collections by their shared prime factors (e.g., detecting
  shared divisors, coprimality structure) under tight performance constraints. It replaces naive divisor scans and heavy per-element
  factorization with input-sensitive precomputation (sieving only up to the actual maximum value), lightweight prime-factor
  tracking, and early exits. Depending on
tags:
- number_theory
- prime_factorization
- sieve
- smallest_prime_factor
- gcd
- coprimality
- frequency_counting
- multiple_counting
- constant_factor_optimization
- input_sensitive_precomputation
triggers:
- You need to determine whether a list of integers shares nontrivial common factors or to classify it by how primes are shared
  between elements.
- Constraints allow values up to around 10^6–10^7 and counts up to around 10^5–10^6, so pairwise gcd checks or O(N^2) strategies
  are impossible.
- There is a full sieve or prime-factor table built up to a hard-coded constant (e.g., 10**6) regardless of the actual maximum
  value in the input.
- A sieve or divisor loop always scans the entire fixed range for every test case, even when the maximum encountered value
  is much smaller.
- Per-number factorization uses trial division starting from 2 with increments of 1, or builds fresh sets/dicts of prime factors
  for every element.
- Prime-usage tracking uses Python sets or dictionaries in the inner loop, even though prime values are bounded and can be
  indexed in an array.
- Divisibility aggregation is done via patterns like arr[d::d] or sum(arr[d::d]) inside an outer loop over d, creating large
  temporary slices.
- Sieve implementations use naive nested Python loops for marking multiples without slice-based marking or parity optimizations.
---

## When to use
- You need to determine whether a list of integers shares nontrivial common factors or to classify it by how primes are shared between elements.
- Constraints allow values up to around 10^6–10^7 and counts up to around 10^5–10^6, so pairwise gcd checks or O(N^2) strategies are impossible.
- There is a full sieve or prime-factor table built up to a hard-coded constant (e.g., 10**6) regardless of the actual maximum value in the input.
- A sieve or divisor loop always scans the entire fixed range for every test case, even when the maximum encountered value is much smaller.
- Per-number factorization uses trial division starting from 2 with increments of 1, or builds fresh sets/dicts of prime factors for every element.
- Prime-usage tracking uses Python sets or dictionaries in the inner loop, even though prime values are bounded and can be indexed in an array.
- Divisibility aggregation is done via patterns like arr[d::d] or sum(arr[d::d]) inside an outer loop over d, creating large temporary slices.
- Sieve implementations use naive nested Python loops for marking multiples without slice-based marking or parity optimizations.

## Steps
1. Derive effective bounds from input rather than constants
2. Choose an aggregation strategy based on constraints
3. • If max_val is up to about 10^6–10^7 and you will factor many numbers, prefer an SPF-based sieve for O(max_val log log max_val) preprocessing and O(log A) factorization per element. • If the number of elements is large but max_val is moderate and you only
4. Implement an efficient sieve or prime-basis precomputation
5. • For SPF: – Allocate an integer array spf[0..max_val]. Initialize spf[i] = 0. – For i from 2 to max_val: * If spf[i] == 0, set spf[i] = i and mark multiples j = i * i, i * (i+1), up to max_val with spf[j] = i if they are still 0. – Use parity optimizations
6. Factor numbers using the chosen representation
7. Track prime usage efficiently and support early exit
8. • Avoid Python sets and dicts in hot loops when indices are bounded; replace them with simple arrays of integers or booleans. • For pairwise/shared-prime detection using factorization: – Maintain an array count_by_prime (or last_seen_index) indexed by prime

## Complexity
- Time: SPF-based approach: O(max_val log log max_val + N * log max_val), where N is the number of input values and max_val is their maximum.
- Space: SPF-based approach: O(max_val) space for the SPF array, plus O(N) for storing input values and O(max_val) for any prime-usage counters.

## Pitfalls
- Keeping a hard-coded sieve limit (like 10**6) instead of adapting it to the maximum value seen in the input, wasting time and memory for easy cases.
- Using list slicing over multiples (arr[d::d]) and summing or scanning them in a loop over all d, which creates O(M log M) temporary elements and leads to heavy allocation and garbage collection.
- Implementing sieves with naive nested Python loops that mark multiples for every i without parity tricks, starting-at-i*i optimizations, or slice-based marking.
- Factoring every number using trial division from 2 up to sqrt(n) with step size 1, even when a prime table or SPF array is already available or cheap to build.
- Creating Python sets or dicts per element to store prime factors, causing many allocations and hash operations in tight loops instead of using simple arrays of counts or flags.
- Computing the same prime factors repeatedly for many numbers instead of aggregating information by primes (e.g., via a frequency array and a scan over divisors).
- Failing to implement early exits: continuing sieve or factorization work after a decisive shared-prime condition has been detected.
- Performing a full scan (like max(counts)) over a large array of length up to max_val when only a small subset of indices has been touched or when the condition could have been checked during updates.

## When not to use
- When the value range max_val is extremely large (e.g., much larger than 10^7–10^8) and cannot be reasonably sieved or indexed in arrays; in such cases, you may need probabilistic primality tests, sparse factorization
- When the number of elements N is very small (e.g., N <= a few hundred), where simple pairwise gcd checks or direct trial division per number is simpler and sufficiently fast without any sieve or global aggregation.
- When you only need to factor a few isolated numbers with potentially large magnitude; using a full-range sieve or frequency over [1..max_val] would be overkill compared to targeted factorization algorithms.
- When working in languages or environments where array-based sieves over the given max_val are infeasible due to memory constraints; in that case, you should not allocate O(max_val) arrays and should consider
- When the classification logic does not depend on prime structure or shared divisors at all (e.g., problems purely about ordering, sums, or non-number-theoretic predicates); using this operator would add unnecessary

## Minimal example
Before:
```py
def classify_shared_primes(arr):
    LIM = 10**6
    spf = list(range(LIM + 1))
    for i in range(2, int(LIM**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, LIM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    prime_buckets = {}
    for x in arr:
        factors = set()
        y = x
        while y > 1:
            p = spf[y]
            factors.add(p)
            while y % p == 0:
                y //= p
        for p in factors:
            prime_buckets.setdefault(p, []).append(x)
    return prime_buckets
```
After:
```py
from math import gcd

def classify_shared_primes(arr):
    g = 0
    for x in arr:
        g = gcd(g, x)
    if g > 1:
        return {g: arr[:]}
    max_val = max(arr)
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            step = i
            for j in range(i * i, max_val + 1, step):
                if spf[j] == j:
                    spf[j] = i
    prime_buckets = {}
    for x in arr:
        y, factors = x, set()
        while y > 1:
            p = spf[y]
            factors.add(p)
            while y % p == 0:
                y //= p
        for p in factors:
            prime_buckets.setdefault(p, []).append(x)
    return prime_buckets
```
