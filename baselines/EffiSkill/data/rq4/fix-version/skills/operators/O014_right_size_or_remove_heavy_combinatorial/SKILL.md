---
skill_id: O014
type: operator
language: python
family: combinatorics
name: Right size or Remove Heavy Combinatorial Precomputation
description: Optimize combinatorics-heavy number-theory code by eliminating or tightening large factorial/inverse-factorial
  precomputations and avoiding unnecessary heavy numeric libraries. Replace fixed-size global tables (often built with NumPy)
  with either (a) small, input-aware factorial tables in pure Python or (b) direct multiplicative binomial formulas when only
  a handful of small-k combinations are needed. Ensure that
tags:
- python
- optimization
- combinatorics
- binomial_coefficient
- factorial_precomputation
- number_theory
- prime_factorization
- modular_arithmetic
- numpy_removal
- constant_factor_reduction
triggers:
- Code precomputes factorial and/or inverse-factorial arrays up to a large fixed MAX_N (e.g., 1e5–1e6) regardless of the actual
  input size or needed combination arguments.
- Global precomputation of factorial tables is executed at import time or before reading input, incurring a large one-time
  cost even for tiny inputs.
- A heavy numeric library (commonly NumPy) is imported solely to build factorial tables or do 1D prefix products for modular
  combinatorics.
- The number of binomial coefficient evaluations is very small (e.g., one per distinct prime factor of an integer), and each
  uses small k (such as prime exponents), but the code still builds large factorial tables.
- Runtime or memory is dominated by factorial/inverse-factorial precomputation, while the actual combinatorial logic and factorization
  are relatively light.
- Precomputation bounds are hard-coded and not obviously derived from N, M, or other input parameters (e.g., always using
  2*10^5 or 10^6).
- Theoretical constraints allow N to be very large (up to 1e9 or more), but the code attempts factorial-table-based nCr limited
  to a much smaller fixed bound.
- Performance profiling or intuition shows that loops over factorial table indices (or NumPy array reshaping and prefix products)
  dominate total runtime.
---

## When to use
- Code precomputes factorial and/or inverse-factorial arrays up to a large fixed MAX_N (e.g., 1e5–1e6) regardless of the actual input size or needed combination arguments.
- Global precomputation of factorial tables is executed at import time or before reading input, incurring a large one-time cost even for tiny inputs.
- A heavy numeric library (commonly NumPy) is imported solely to build factorial tables or do 1D prefix products for modular combinatorics.
- The number of binomial coefficient evaluations is very small (e.g., one per distinct prime factor of an integer), and each uses small k (such as prime exponents), but the code still builds large factorial tables.
- Runtime or memory is dominated by factorial/inverse-factorial precomputation, while the actual combinatorial logic and factorization are relatively light.
- Precomputation bounds are hard-coded and not obviously derived from N, M, or other input parameters (e.g., always using 2*10^5 or 10^6).
- Theoretical constraints allow N to be very large (up to 1e9 or more), but the code attempts factorial-table-based nCr limited to a much smaller fixed bound.
- Performance profiling or intuition shows that loops over factorial table indices (or NumPy array reshaping and prefix products) dominate total runtime.

## Steps
1. Analyze actual combinatorial needs: 1, e)).
2. Analyze actual combinatorial needs: Determine the true maximum n involved (e.g., max over N + max_exponent) and typical k (often small exponents).
3. Decide between two strategies: Strategy A (adaptive tables): if many nCr calls or k can be large, keep factorial/inverse-factorial tables but size them tightly to the required maximum n.
4. Decide between two strategies: Strategy B (on-the-fly binomials): if the number of nCr calls is tiny and each has small k, drop tables entirely and use a direct O(k) multiplicative nCr implementation.
5. If using adaptive factorial tables (Strategy A): Compute limit = required_max_n + safety_margin (small constant like 50–100 based on exponent bounds).
6. If using adaptive factorial tables (Strategy A): Allocate fact = [1] * (limit + 1) and inv_fact = [1] * (limit + 1).
7. If using adaptive factorial tables (Strategy A): Fill fact in a single forward loop: for i in range(1, limit + 1): fact[i] = fact[i-1] * i % MOD.
8. If using adaptive factorial tables (Strategy A): Compute inv_fact[limit] with pow(fact[limit], MOD-2, MOD) (for prime MOD), then fill inv_fact backwards: for i in range(limit, 0, -1): inv_fact[i-1] = inv_fact[i] * i % MOD.

## Complexity
- Time: (pattern dependent)
- Space: Auxiliary space is reduced from O(F) for large global factorial/inverse-factorial arrays (often with additional NumPy buffers) to O(limit) with limit tied to actual needs, or to O(1)–O(log M) when using only direct multiplicative

## Pitfalls
- Leaving hard-coded large precomputation bounds in place (e.g., still computing up to 1e6) even after introducing input-aware sizing, negating most benefits.
- Switching to direct multiplicative nCr for cases where k is not actually small, leading to O(k) per query overhead that can be worse than a well-sized factorial table.
- Underestimating the maximum n used in combinations when sizing factorial tables, causing out-of-range accesses or silent logic errors.
- Incorrect treatment of modular division in multiplicative nCr: forgetting to use modular inverses under a prime modulus, or applying modulo before division so the division is no longer exact.
- Applying this optimization in a multi-query setting with large N and many nCr evaluations without amortizing precomputation, causing repeated expensive recomputation of factorial tables.
- Removing NumPy but accidentally increasing Python-level loops (e.g., computing modular inverses one-by-one repetitively instead of using a backward sweep), leading to unnecessary overhead.
- Keeping heavy global work at import time (large precomputation before reading inputs), which still penalizes small test cases and harms interactive or library use.
- Not exploiting symmetry in nCr (using k instead of min(k, n-k)) in multiplicative implementations, roughly doubling loop lengths for some arguments.

## When not to use
- Problems that require a very large number of binomial coefficient evaluations over a wide range of n and k (e.g., DP with many states), where a single O(limit) factorial/inverse-factorial precomputation amortized
- Environments where NumPy (or similar) is known to be fast, allowed, and precomputation ranges are genuinely large and input-dependent (e.g., many combinations up to n ≈ 10^6), making vectorized factorial precomputation
- Scenarios where N and k can both be large and there are many queries, so direct O(k) multiplicative nCr per query would be too slow compared to O(1) table-based nCr.
- Cases where the existing precomputation bound is already tight and input-aware, and factorial tables are reused heavily across multiple test cases or multiple independent computations.
- Situations in which numerical stability or exact integer results without modular arithmetic are required for very large n and k; in such settings, combinatorial libraries or specialized algorithms may be more

## Minimal example
Before:
```py
MOD = 10**9 + 7
MAXN = 10**6
fact = [1] * (MAXN + 1)
for i in range(1, MAXN + 1):
    fact[i] = fact[i-1] * i % MOD

def ways_from_exponents(n, exps):
    # product of C(n + e - 1, e)
    res = 1
    for e in exps:
        num = fact[n + e - 1]
        den = pow(fact[n-1] * fact[e] % MOD, MOD-2, MOD)
        res = res * (num * den % MOD) % MOD
    return res
```
After:
```py
MOD = 10**9 + 7

def nCr_small_k(n, k):  # O(k), good when exponents are small
    k = min(k, n - k)
    num = den = 1
    for i in range(1, k + 1):
        num = num * (n - k + i) % MOD
        den = den * i % MOD
    return num * pow(den, MOD-2, MOD) % MOD

def ways_from_exponents(n, exps):  # no giant global factorial table
    res = 1
    for e in exps:
        res = res * nCr_small_k(n + e - 1, e) % MOD
    return res
```
