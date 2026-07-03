---
skill_id: O023
type: operator
language: python
family: combinatorics
name: Modular Binomial Combinatorics Without Big Int Factorials
description: Optimize combinatorial computations under a modulus by eliminating arbitrary-precision factorials and integer-division-based
  binomial formulas. Instead, precompute factorials and inverse factorials modulo a prime (or use multiplicative binomial
  formulas over the small parameter), and apply modular reduction at every step. This shifts cost from superlinear big-integer
  arithmetic to linear-time precomputation plus
tags:
- combinatorics
- binomial_coefficient
- modular_arithmetic
- number_theory
- prime_factorization
- precomputation
- performance
- big_integer_avoidance
triggers:
- Binomial coefficients or combinatorial expressions implemented as factorial(n) // (factorial(k) * factorial(n-k)) with only
  a final % MOD.
- Use of math.factorial (or equivalent big-int factorial) on arguments as large as O(1e5) or larger when only results modulo
  a fixed prime are required.
- Patterns like products of consecutive integers divided by factorial(e) (e.g. (N+e-1)*(N+e-2)* / e!), which are equivalent
  to C(N+e-1, e).
- Stars-and-bars or repeated combination formulas C(N+e-1, e) after prime factorization of some M, especially with N up to
  ~1e5 and M up to ~1e9.
- TLE/MLE or visible performance issues despite loop counts looking linear, with profiling showing factorial or big-int division
  dominating.
- Intermediate combinatorial values growing to huge integers (many thousands of digits) before any modular reduction.
- Presence of pow(x, MOD-2, MOD) where x is built from big factorials or large integer products instead of already-modular
  values.
- Repeated factorial calls on related arguments (n, n-1, n+e, k, n-k) instead of a shared factorial table.
---

## When to use
- Binomial coefficients or combinatorial expressions implemented as factorial(n) // (factorial(k) * factorial(n-k)) with only a final % MOD.
- Use of math.factorial (or equivalent big-int factorial) on arguments as large as O(1e5) or larger when only results modulo a fixed prime are required.
- Patterns like products of consecutive integers divided by factorial(e) (e.g. (N+e-1)*(N+e-2)* / e!), which are equivalent to C(N+e-1, e).
- Stars-and-bars or repeated combination formulas C(N+e-1, e) after prime factorization of some M, especially with N up to ~1e5 and M up to ~1e9.
- TLE/MLE or visible performance issues despite loop counts looking linear, with profiling showing factorial or big-int division dominating.
- Intermediate combinatorial values growing to huge integers (many thousands of digits) before any modular reduction.
- Presence of pow(x, MOD-2, MOD) where x is built from big factorials or large integer products instead of already-modular values.
- Repeated factorial calls on related arguments (n, n-1, n+e, k, n-k) instead of a shared factorial table.

## Steps
1. Recognize combinatorial structure
2. - Identify expressions of the form factorial(n) // (factorial(k) * factorial(n-k)), products of consecutive integers divided by factorial(e), or repeated stars-and-bars terms like C(N+e-1, e).
3. - Confirm that the final required answer is modulo a prime MOD (e.g., 1_000_000_007), and that exact big integers are not needed.
4. Remove unnecessary big-int work
5. - Delete any computation of large factorials whose results are not used in the final formula (e.g., factorial(N-1) that is never referenced).
6. - Replace caches of big factorial integers (dict-based factorial caching) with an index-based modular precomputation strategy.
7. Choose computation strategy
8. - If n can be up to ~1e5 and you may need many different nCr values: use factorial and inverse-factorial precomputation modulo MOD.

## Complexity
- Time: With factorial precomputation: O(max_n + sqrt(M) + K) where max_n is the largest n in any binomial (e.g., N + max_exponent), M is any factored integer, and K is the number of binomial queries (each O(1)). With multiplicative formulas over
- Space: O(max_n) machine-word integers for factorial and inverse-factorial tables when using precomputation, plus O(number_of_primes) for factorization state. Using only multiplicative binomials over small k requires O(1) additional space beyond

## Pitfalls
- Forgetting to size the factorial tables large enough (max_n too small), causing index errors or incorrect nCr results when n > max_n.
- Using factorial and inverse factorial precomputation when n itself can be extremely large (e.g., up to 1e12 or higher), making max_n infeasible; in such cases an O(k) multiplicative formula over small k is needed
- Assuming MOD is prime and using pow(x, MOD-2, MOD) for modular inverses when MOD is actually composite, which breaks Fermat-based inverses.
- Applying the modulo only at the very end of a large product, causing intermediate Python integers to grow huge and negate the performance benefits.
- Neglecting the k = min(k, n-k) symmetry, leading to unnecessary extra multiplications and slower binomial computations.
- Mixing exact integer division with modular arithmetic (e.g., building a big numerator and doing `// factorial(k)` instead of using modular inverses), which restores big-int costs.
- Recomputing factorial tables for each query instead of reusing a single precomputation when multiple combinations are needed.
- Incorrect bounds for max_n in stars-and-bars patterns (e.g., needing up to N + max_exponent - 1 but only precomputing to N).

## When not to use
- When no modulus is involved and exact combinatorial values are required; in that case, big integers and factorial-based formulas may be appropriate.
- When the modulus is not prime and you cannot easily adapt to a modular inverse strategy (e.g., via extended GCD or factorial decomposition); Fermat-style inverse and the standard fact/inv_fact pattern may not apply
- When n in binomial coefficients can be extremely large (e.g., up to 1e12 or more) and k is not small; precomputing factorials up to n is infeasible, and even an O(k) multiplicative formula may be too slow if k is large.
- When the number of distinct binomial computations is tiny and parameters are small (e.g., n ≤ 20); the overhead of setting up global factorial tables can outweigh any benefit over straightforward direct computation.
- When memory is extremely constrained and the required max_n for factorial tables is very large; the O(max_n) space for fact and inv_fact may be unacceptable.
- When the primary bottleneck is elsewhere (e.g., extremely expensive factorization or heavy non-numeric processing) and combinatorial terms are provably negligible in cost; optimizing binomial computation then yields

## Minimal example
Before:
```py
import math
MOD = 1_000_000_007

def ways(n, m):  # stars-and-bars: C(n + m - 1, n - 1)
    return math.factorial(n + m - 1) // (math.factorial(n - 1) * math.factorial(m)) % MOD

distributions = [(50_000, 20_000), (100_000, 10_000)]
ans = 1
for n, m in distributions:
    ans = ans * ways(n, m) % MOD
```
After:
```py
MOD = 1_000_000_007
max_n = 120_000  # >= max(n + m - 1)
fact = [1] * (max_n + 1); inv = [1] * (max_n + 1)
for i in range(1, max_n + 1): fact[i] = fact[i - 1] * i % MOD
inv[max_n] = pow(fact[max_n], MOD - 2, MOD)
for i in range(max_n, 0, -1): inv[i - 1] = inv[i] * i % MOD
nCr = lambda n, r: fact[n] * inv[r] % MOD * inv[n - r] % MOD
distributions = [(50_000, 20_000), (100_000, 10_000)]
ans = 1
for n, m in distributions:  # C(n + m - 1, n - 1) mod MOD via precomputed tables
    ans = ans * nCr(n + m - 1, n - 1) % MOD
```
