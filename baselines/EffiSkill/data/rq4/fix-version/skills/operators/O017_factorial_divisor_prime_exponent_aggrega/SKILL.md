---
skill_id: O017
type: operator
language: python
family: combinatorics
name: Factorial Divisor Prime Exponent Aggregation Optimization
description: 'Optimize computations that depend on the prime exponents of a product of consecutive integers (often n!) by
  (1) avoiding per-integer re-factorization and per-prefix primality recomputation, and (2) choosing an aggregation strategy
  that balances asymptotic complexity with language-level overhead. Typical efficient patterns: a single sieve plus Legendre-style
  exponent summation; or a single global factorization of'
tags:
- number_theory
- prime_factorization
- factorial_like_product
- divisor_counting
- prime_sieve
- modular_arithmetic
- constant_factor_optimization
- python_overhead
triggers:
- Goal is to count divisors or otherwise use the prime factorization of a product 1·2· ·n or a similar consecutive product.
- Code factors each integer k in [1..n] separately and aggregates prime exponents (e.g., loops over k and runs a factorization
  helper for each).
- A primality or sieve routine is called inside a loop over k, recomputing primes up to k for every k.
- Exponent bookkeeping uses length-n arrays or per-k vectors that are merged elementwise in an outer loop (O(n^2) array work).
- Factorization uses trial division up to sqrt(k) for every k without any global sieve or shared prime information.
- Large lists of factors are collected and then passed to collections.Counter only to get multiplicities.
- A big factorial is recomputed multiple times (e.g., math.factorial(n) inside a loop) or n! is factored repeatedly.
- Prime sieve arrays are fixed-size (e.g., length 10000) but indexed up to a variable n, hinting at scaling or correctness
  issues.
---

## When to use
- Goal is to count divisors or otherwise use the prime factorization of a product 1·2· ·n or a similar consecutive product.
- Code factors each integer k in [1..n] separately and aggregates prime exponents (e.g., loops over k and runs a factorization helper for each).
- A primality or sieve routine is called inside a loop over k, recomputing primes up to k for every k.
- Exponent bookkeeping uses length-n arrays or per-k vectors that are merged elementwise in an outer loop (O(n^2) array work).
- Factorization uses trial division up to sqrt(k) for every k without any global sieve or shared prime information.
- Large lists of factors are collected and then passed to collections.Counter only to get multiplicities.
- A big factorial is recomputed multiple times (e.g., math.factorial(n) inside a loop) or n! is factored repeatedly.
- Prime sieve arrays are fixed-size (e.g., length 10000) but indexed up to a variable n, hinting at scaling or correctness issues.

## Steps
1. Identify structure: recognize that the quantity depends only on prime exponents of a product of consecutive integers (often n!), so the final result can be expressed as a product over primes p of a simple function of e_p (commonly (e_p + 1)).
2. Eliminate per-k primality recomputation: replace any primality test inside a loop over k with a single sieve of Eratosthenes (or smallest-prime-factor sieve) up to n, built once.
3. Avoid per-k exponent vectors: instead of returning length-n arrays or dicts per integer k and merging them, maintain a single global accumulator of prime exponents (array indexed by value or compact dict over primes).
4. Choose an aggregation strategy
5. • Strategy A (sieve + Legendre): generate all primes p ≤ n by sieve; for each prime, compute e_p using n // p + n // p^2 + until p^k > n; accumulate the result (e_p + 1) into an answer under the modulus.
6. • Strategy B (per-number factorization with shared state): for k in [2..n], factor k once (trial division up to sqrt(k) or using precomputed smallest prime factors), and increment global exponent counters for each encountered prime; at the end, compute the
7. • Strategy C (single factorial factorization, language-specific): compute n! once using an optimized library (e.g., math.factorial), then factor this single big integer using trial division by primes or small integers, counting exponents on the fly; finally
8. During factorization, count exponents directly: in the trial-division loop, maintain a local exponent counter for the current prime and add it to the global accumulator, instead of appending factors to a list and using Counter later.

## Complexity
- Time: (pattern dependent)
- Space: Sieve + Legendre exponents: O(n) space for the sieve array and O(π(n)) for the prime list.

## Pitfalls
- Recomputing primality or a sieve for every prefix 1..k instead of a single sieve up to n, causing super-quadratic time.
- Factoring every integer 1..n separately and building per-k exponent vectors, then merging them with O(n^2) list operations.
- Using Counter on a factor list produced in a tight loop instead of counting exponents directly during trial division.
- Heavy use of defaultdict, dict, and list(dict.items()) inside core loops, increasing constant factors and allocation churn.
- Using naive trial division up to sqrt(k) for each k with no precomputed primes, leading to ≈O(n^{3/2}) total operations.
- Computing math.factorial(n) repeatedly inside a loop or dividing a fresh copy of n! for each prime, rather than computing and factoring n! once.
- Allocating fixed-size sieve arrays (e.g., length 10000) independent of n, leading to index errors or artificial caps on input size.
- Applying modulus operations more often than needed (e.g., on per-prime exponent contributions or inside tight inner loops) instead of once per multiplication in the final aggregation.

## When not to use
- Do not explicitly compute and factor n! (Strategy C) when n can be large enough that n! no longer fits comfortably in memory or big-int operations are too slow (e.g., n much larger than ≈10^4). Prefer sieve + Legendre
- Avoid per-number trial-division aggregation (Strategy B) when n is very large (e.g., ≥10^6) and a sieve-based exponent method (Strategy A) is available; the O(n^{3/2}) factorization time will likely be too slow.
- Do not introduce dictionary-heavy or Counter-based pipelines in an attempt to improve asymptotic complexity if constraints are small and interpreter overhead will dominate; simple array-based or direct loop approaches
- Avoid fixed-size sieve arrays tied to assumed limits when the actual n is input-dependent; use dynamic sizing instead to maintain correctness and proper complexity.
- Avoid highly specialized floor-sum shortcuts when the problem requires full prime exponent distributions or more detailed factorization than simple divisor counts.

## Minimal example
Before:
```py
from collections import Counter

def num_divisors_factorial(n):
    factors = []
    for k in range(2, n + 1):            # re-factor each k separately
        x = k
        d = 2
        while d * d <= x:               # repeated trial division
            while x % d == 0:
                factors.append(d)
                x //= d
            d += 1
        if x > 1:
            factors.append(x)
    exponents = Counter(factors)        # aggregate after building big list
    ans = 1
    for e in exponents.values():
        ans *= (e + 1)
    return ans
```
After:
```py
def sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0:2] = [False, False]
    for p in range(2, int(limit ** 0.5) + 1):
        if is_prime[p]:
            step = p * p
            is_prime[step:limit + 1:p] = [False] * ((limit - step) // p + 1)
    return [i for i, v in enumerate(is_prime) if v]

def num_divisors_factorial(n):
    ans = 1
    for p in sieve_primes(n):           # use Legendre formula on primes only
        exp = 0
        m = n
        while m:
            m //= p
            exp += m                    # exponent of p in n!
        ans *= (exp + 1)
    return ans
```
