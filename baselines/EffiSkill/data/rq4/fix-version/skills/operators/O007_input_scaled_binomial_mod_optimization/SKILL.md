---
skill_id: O007
type: operator
language: python
family: combinatorics
name: Input Scaled Binomial Mod Optimization
description: Reusable optimization pattern distilled from weighted traces.
tags:
- combinatorics
- binomial_coefficient
- modular_arithmetic
- precomputation
- space_optimization
- time_optimization
- single_query
- number_theory
triggers:
- Factorial, inverse, or inverse-factorial arrays are precomputed up to a large fixed constant (e.g., 1e6 or 2e5) that is
  not derived from the current input.
- Only a single or very small number of nCk evaluations are performed, yet O(N) factorial/inverse tables are built where N
  is close to a worst-case bound.
- An inverse table (or multiple inverse tables) is built linearly (sieve-style) when the code ultimately uses only a handful
  of modular inverses.
- A large DP or Pascal-triangle table is constructed to compute a single binomial coefficient.
- Tight memory or time limits with constraints where N (from coordinates, counts, or factorization exponents) can be large
  enough that three O(N) arrays are costly.
- Combinations needed only along a single row (e.g., C(n-1, k) for k in a small range) or as a simple product over a small
  set (e.g., per prime factor), suggesting incremental or multiplicative formulas.
---

## When to use
- Factorial, inverse, or inverse-factorial arrays are precomputed up to a large fixed constant (e.g., 1e6 or 2e5) that is not derived from the current input.
- Only a single or very small number of nCk evaluations are performed, yet O(N) factorial/inverse tables are built where N is close to a worst-case bound.
- An inverse table (or multiple inverse tables) is built linearly (sieve-style) when the code ultimately uses only a handful of modular inverses.
- A large DP or Pascal-triangle table is constructed to compute a single binomial coefficient.
- Tight memory or time limits with constraints where N (from coordinates, counts, or factorization exponents) can be large enough that three O(N) arrays are costly.
- Combinations needed only along a single row (e.g., C(n-1, k) for k in a small range) or as a simple product over a small set (e.g., per prime factor), suggesting incremental or multiplicative formulas.

## Steps
1. Analyze required binomial range: Identify all n, k pairs actually used (from grid transforms, stars-and-bars formulas, or factorization exponents).
2. Analyze required binomial range: Compute an exact or safe upper bound U on n (e.g., U = (x + y) / 3, or U = N + max_exponent).
3. Decide strategy based on query count: If only O(1) or very few nCk evaluations are needed, prefer an on-the-fly multiplicative formula with pow-based modular inverse (no tables).
4. Decide strategy based on query count: If many nCk values are needed (e.g., across a row or many test cases), keep factorial-based tables but size them to U instead of a fixed global constant.
5. 3A. On-the-fly multiplicative binomial (no tables): k).
6. 3A. On-the-fly multiplicative binomial (no tables): Initialize num = 1, den = 1.
7. 3A. On-the-fly multiplicative binomial (no tables): For i in 1..k
8. 3A. On-the-fly multiplicative binomial (no tables): i) % MOD

## Complexity
- Time: Typical improvement is from O(N_fixed + N_use) to O(N_use) or O(k + log MOD), where N_fixed is a large hard-coded bound, N_use is the true maximum n in nCk, and k = min(n, k). For single or few binomial queries, factorial-based O(N_use)
- Space: Reduced from O(N_fixed) or O(N_use) tables (often multiple arrays) to either O(N_use) with a single factorial or (factorial + inverse-factorial) pair, or down to O(1) additional space for purely multiplicative nCk. This is especially

## Pitfalls
- Keeping a fixed global precomputation bound that is much larger than any actual n, wasting time and memory on every run.
- Choosing factorial-based tables when there is only a single nCk query; the amortization never pays off and is slower than a simple multiplicative formula.
- Hard-coding precomputation limits that are too small, causing out-of-range factorial access or silent wrong answers when input grows.
- Building full modular inverse tables (or both inv[] and inv_fact[]) when only factorials plus a few pow-based inverses are needed.
- Forgetting to reduce products modulo MOD in intermediate factorial or combinatorial computations, leading to very large Python integers and slow arithmetic.
- Using floating-point division and equality checks to derive integer parameters (e.g., coordinates -> moves), which is slower and can be unsafe for large integers.
- Recomputing pow(base, MOD-2, MOD) inside loops for the same base instead of caching it once.
- Allocating 2D DP tables or large 1D arrays for Pascal-style binomial DP when a direct formula would be more efficient.

## When not to use
- When the application performs many nCk queries across a wide range of n and k (e.g., thousands of queries per test) and a well-sized factorial/inverse-factorial table is already amortized and fits comfortably in memory.
- When inputs are tiny (e.g., n ≤ 1000) and existing code using full tables or DP is already fast and clear; micro-optimizing precomputation or switching to multiplicative formulas may not justify the added complexity.
- When the modulus is not prime and Fermat’s little theorem does not apply; alternative inverse strategies (e.g., extended GCD, Lucas theorem, or prime-power-specific methods) may be more appropriate than this pattern.
- When combinatorial logic genuinely requires entire rows or large parts of Pascal’s triangle (true DP dependence among many states), not just a few isolated nCk values.

## Minimal example
Before:
```py
MOD = 10**9 + 7
MAXN = 10**6 + 5
fact = [1] * MAXN
for i in range(1, MAXN):
    fact[i] = fact[i - 1] * i % MOD

def nCk(n, k):
    if k < 0 or k > n: return 0
    inv = pow(fact[k] * fact[n - k] % MOD, MOD - 2, MOD)
    return fact[n] * inv % MOD
```
After:
```py
MOD = 10**9 + 7

def nCk(n, k):
    if k < 0 or k > n: return 0
    k = min(k, n - k)
    num = den = 1
    for i in range(1, k + 1):
        num = num * (n - k + i) % MOD
        den = den * i % MOD
    return num * pow(den, MOD - 2, MOD) % MOD
```
