---
skill_id: O015
type: operator
language: python
family: combinatorics
name: Optimize Modular Combinatorics And Binomials
description: Reusable optimization pattern distilled from weighted traces.
tags:
- combinatorics
- binomial_coefficient
- modular_arithmetic
- number_theory
- prime_factorization
- optimization
- precomputation
- performance
- Python
triggers:
- A helper computing C(n, k) or a similar combinatorial term by looping i from 1..k and calling pow(i, MOD-2, MOD) in every
  iteration.
- A binomial-like recurrence in a loop, such as curr *= (n - i) * inv(i) % MOD, where inv(i) is recomputed each time via pow.
- A loop that linearly searches over an integer parameter (e.g., counts of move types) to satisfy a simple linear system,
  where the variables appear only linearly and can be solved algebraically.
- A combination computed as C(n, k) with k taken directly from a large input (like N-1 or up to 1e5–1e9) even though the complementary
  parameter n-k is much smaller (e.g., a prime exponent or difference of coordinates).
- Factorial and inverse-factorial arrays sized up to a large N even though only C(n, i) for i ≤ K or a handful of binomials
  are ever used.
- Prime factorization of an integer M followed by loops that, for each prime factor, iterate up to N (or N-1) to compute C(N+e-1,
  N-1) using modular inverses, where e is the exponent of the prime.
- Repeated recomputation of powers like base^(n-1-i) via pow(base, exponent, MOD) inside a loop over i, rather than maintaining
  a running power.
- Presence of custom modular exponentiation implementations in Python used for modular inverses or powers inside hot loops,
  instead of built-in pow(base, exp, mod).
---

## When to use
- A helper computing C(n, k) or a similar combinatorial term by looping i from 1..k and calling pow(i, MOD-2, MOD) in every iteration.
- A binomial-like recurrence in a loop, such as curr *= (n - i) * inv(i) % MOD, where inv(i) is recomputed each time via pow.
- A loop that linearly searches over an integer parameter (e.g., counts of move types) to satisfy a simple linear system, where the variables appear only linearly and can be solved algebraically.
- A combination computed as C(n, k) with k taken directly from a large input (like N-1 or up to 1e5–1e9) even though the complementary parameter n-k is much smaller (e.g., a prime exponent or difference of coordinates).
- Factorial and inverse-factorial arrays sized up to a large N even though only C(n, i) for i ≤ K or a handful of binomials are ever used.
- Prime factorization of an integer M followed by loops that, for each prime factor, iterate up to N (or N-1) to compute C(N+e-1, N-1) using modular inverses, where e is the exponent of the prime.
- Repeated recomputation of powers like base^(n-1-i) via pow(base, exponent, MOD) inside a loop over i, rather than maintaining a running power.
- Presence of custom modular exponentiation implementations in Python used for modular inverses or powers inside hot loops, instead of built-in pow(base, exp, mod).

## Steps
1. Identify the exact combinatorial quantity: For problems involving prime factorization, check if each prime p^e contributes a term of the form C(N+e-1, e) (stars and bars pattern).
2. Solve algebraic parameters instead of searching: When a loop searches over integer variables to satisfy a linear system (e.g., combinations of two move types to reach coordinates), derive closed-form expressions for those counts using algebra.
3. Solve algebraic parameters instead of searching: Add early feasibility checks (e.g., divisibility conditions like (x+y) % 3 == 0, non-negativity of counts) and return 0 immediately when impossible.
4. Minimize binomial loop length using symmetry: Rewrite C(n, k) as C(n, min(k, n-k)) to reduce iteration length when using multiplicative formulas.
5. Minimize binomial loop length using symmetry: In factorization-based formulas, prefer C(N+e-1, e) over C(N+e-1, N-1) since e (prime exponent) is O(log M) and much smaller than N.
6. Minimize binomial loop length using symmetry: Apply this systematically in any nCk helper before looping.
7. Choose an appropriate binomial computation strategy
8. 4a. On-the-fly multiplicative formula (no large precomputation): Use when k is relatively small (e.g., k <= 2e5, or k ≈ sum of small exponents) and you only need a few binomial coefficients.

## Complexity
- Time: Typical optimized patterns achieve: (1) for grid/path-style problems: O(n) to build factorials/inverses and O(1) per binomial query, or O(k + log MOD) using incremental combinations and a single pow; (2) for modular sums like Σ C(n-1, i)
- Space: (pattern dependent)

## Pitfalls
- Leaving pow( , MOD-2, MOD) inside a hot loop after other refactors, which keeps an unnecessary O(log MOD) factor per iteration.
- Using C(n, k) directly from inputs without applying k = min(k, n-k), leading to loops over the larger side of the binomial and potential timeouts.
- Precomputing factorials or inverse factorials up to a massive N when only a small subset of binomials or indices is needed, causing excessive memory use and startup time.
- Switching to integer-based binomial computation (num // den) without ensuring exact divisibility, incorrect integer types, or failing to reduce the result modulo MOD afterward.
- Incorrectly deriving or applying algebraic formulas for move counts (e.g., mis-solving a small linear system), which yields negative or non-integer parameters and wrong combinatorial input.
- Factoring an integer M but still using N as the loop bound in combination routines, so overall cost is O(#primes * N) instead of O(#primes * exponent).
- Using Python recursion for modular exponentiation or recursive nCk when iteration with built-in pow or iterative formulas would avoid recursion overhead and recursion limits.
- Forgetting to clamp or validate k in nCk helpers (e.g., allowing k < 0 or k > n), which can cause index errors when using factorial tables.

## When not to use
- When n and k are extremely small (e.g., <= 50) and performance is not critical; simple direct formulas may be clearer than introducing precomputation or complex recurrences.
- When the modulus is not prime or modular inverses are not always defined; techniques relying on pow(x, MOD-2, MOD) or inverse tables assume a prime modulus and multiplicative inverses for all 1 <= x < MOD.
- When computing combinatorics without a modulus and exact large integers are required throughout; using modulo arithmetic or Fermat-based inverses may not apply.
- When the problem already provides strict upper bounds that make factorial precomputation trivial (e.g., n <= 2000) and memory is abundant; the constant-factor improvements from more intricate recurrences may not
- When using languages or environments where built-in big integers are not available or are very slow, making integer-based num // den binomial computations less attractive compared to modular-inverse methods.
- When your algorithm requires many different moduli or dynamically changing moduli; precomputations of factorials or inverse tables tied to a single MOD may not be reusable and may cost more than they save.

## Minimal example
Before:
```py
MOD = 10**9 + 7

def ways_stars_bars(n, k):  # C(n+k-1, k) mod MOD
    res = 1
    for i in range(1, k + 1):
        res = res * (n + k - i) % MOD * pow(i, MOD - 2, MOD) % MOD
    return res
```
After:
```py
MOD = 10**9 + 7

_fact = [1]
_ifact = [1]

def _ensure_fact(m):
    cur = len(_fact)
    if cur <= m:
        for i in range(cur, m + 1):
            _fact.append(_fact[-1] * i % MOD)
        _ifact.extend([0] * (m + 1 - len(_ifact)))
        _ifact[m] = pow(_fact[m], MOD - 2, MOD)
        for i in range(m, cur, -1):
            _ifact[i - 1] = _ifact[i] * i % MOD

def ways_stars_bars(n, k):  # C(n+k-1, k) mod MOD
    m = n + k - 1
    _ensure_fact(m)
    return _fact[m] * _ifact[k] % MOD * _ifact[m - k] % MOD
```
