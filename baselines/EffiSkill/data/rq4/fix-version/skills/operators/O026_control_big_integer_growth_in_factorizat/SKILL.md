---
skill_id: O026
type: operator
language: python
family: combinatorics
name: Control Big Integer Growth in Factorization Based Combinatorics
description: Optimize number-theoretic and combinatorial code that mixes prime factorization with binomial or factorial expressions
  by eliminating unnecessary big-integer factorials, computing combinations in multiplicative or modular form, and applying
  the modulus early to keep intermediates small. The operator replaces naive n! and factorial-based nCr with exponent-based
  reasoning (e.g., Legendre-style exponent accumulation)
tags:
- number_theory
- combinatorics
- prime_factorization
- binomial_coefficient
- factorial
- modular_arithmetic
- stars_and_bars
- divisor_count
- big_integer_optimization
- precomputation
triggers:
- Code calls factorial on large arguments derived from input (e.g., n, n-1, n+e, up to around 1e5 or larger).
- Binomial coefficients are computed as n! // ((n-k)! * k!) using full-precision integers, with modulus applied only once
  at the end.
- Intermediate products for combinatorial expressions are built as exact big integers and only reduced modulo at the final
  print.
- A caching dictionary of factorial values grows with n, storing many huge integers to speed up repeated factorial calls.
- Prime factorization is followed by loops that construct large falling factorial products proportional to n, even though
  only small exponents e from the factorization are needed.
- Profiling shows most time inside big-integer routines (factorial, multiplication, division, pow for inverses) rather than
  in O(√M) factorization or simple loops.
- Runtime and memory blow up as the first parameter (often denoted n) increases, despite small second parameter (the number
  being factorized).
- Factorization-based formulas for divisor counts or distribution of prime exponents are implemented by first forming a huge
  integer (like n! or a product of many numbers) and then factoring it.
---

## When to use
- Code calls factorial on large arguments derived from input (e.g., n, n-1, n+e, up to around 1e5 or larger).
- Binomial coefficients are computed as n! // ((n-k)! * k!) using full-precision integers, with modulus applied only once at the end.
- Intermediate products for combinatorial expressions are built as exact big integers and only reduced modulo at the final print.
- A caching dictionary of factorial values grows with n, storing many huge integers to speed up repeated factorial calls.
- Prime factorization is followed by loops that construct large falling factorial products proportional to n, even though only small exponents e from the factorization are needed.
- Profiling shows most time inside big-integer routines (factorial, multiplication, division, pow for inverses) rather than in O(√M) factorization or simple loops.
- Runtime and memory blow up as the first parameter (often denoted n) increases, despite small second parameter (the number being factorized).
- Factorization-based formulas for divisor counts or distribution of prime exponents are implemented by first forming a huge integer (like n! or a product of many numbers) and then factoring it.

## Steps
1. Identify the mathematical structure: derive the target quantity in terms of prime exponents and small binomial factors, e.g., divisor counts via exponents e_p(n!), or products over primes of C(N+e-1, e) arising from distributing exponents.
2. Avoid constructing huge composite integers: do not explicitly compute n!, factorial(N-1), or factorials with arguments proportional to the large input; instead, work directly with exponents or small loops.
3. When many binomial evaluations share n, consider modular precomputation: build factorial and inverse-factorial arrays modulo MOD once and answer each C(n, k) in O(1) using `fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD`.
4. Localize big-integer work: if the modulus must be applied only after exact binomial computation, keep big-int usage confined inside the binomial helper; do not multiply many such values together as exact integers.
5. Move modular reduction into the accumulation loop: maintain the global result as `ans = ans * term % MOD` after each factor contribution, rather than taking `% MOD` only once at the end. This bounds the accumulator size.
6. Apply modular reduction selectively inside combinatorial routines if safe (e.g., when using multiplicative formulas with modular inverses), so that all arithmetic on n and k is done with machine-word-sized integers modulo MOD.
7. Refactor prime factorization to minimize overhead: Use trial division with a dynamic bound `d*d <= current_n`, shrinking as factors are removed.
8. Refactor prime factorization to minimize overhead: After handling factor 2, increment d by 2 to skip evens.

## Complexity
- Time: (pattern dependent)
- Space: Reduces space from storing huge factorial big integers (Θ(n log n) bits or large factorial caches) to O(#primes(M) + log M) for factorization and at most O(U) for optional modular factorial precomputation, with all frequently updated

## Pitfalls
- Computing large factorials like factorial(n), factorial(n-1), or factorial(n+e) when n can be up to ~1e5 or larger, causing huge big-int objects and superlinear bit-complexity.
- Using factorial-based nCr directly and assuming math.factorial is fast enough for contest-sized inputs without considering big-int growth.
- Applying the modulus only in the final output expression, letting the global accumulator grow to the product of all combinatorial terms.
- Over-optimizing with factorial caches of exact integers instead of changing the algorithm to avoid large factorials altogether.
- Rewriting a correct modular-precomputation solution (factorials and inverse factorials mod MOD) into a big-integer factorial version for perceived simplicity, drastically worsening performance.
- Factoring enormous composite numbers like n! by trial division rather than computing prime exponents analytically, turning a nearly optimal algorithm into an intractable one.
- Leaving recursive implementations of binomial formulas in place for repeated calls, incurring Python recursion overhead and extra big-int multiplications/divisions.
- Using fixed trial-division bounds based on sqrt(initial_M) instead of a dynamic `d*d <= current_M`, leading to many unnecessary divisor checks after the number has been heavily reduced.

## When not to use
- When input sizes are so small that big-integer factorial and naive binomial computations are clearly within time and memory limits; simplifying the code may be more valuable than optimizing constants.
- When the problem requires exact large combinatorial values (not modulo) and these values are inherently huge; in such cases big-int growth is unavoidable and must be managed rather than eliminated.
- When a robust, precomputed modular combinatorics framework is already in place and proven fast enough; replacing it with ad-hoc multiplicative loops may complicate the codebase without meaningful gains.
- When prime factorization via trial division up to sqrt(M) is itself too slow for the given constraints (e.g., M near 10^12+ with many test cases); in that setting, optimizing factorials is secondary to choosing a

## Minimal example
Before:
```py
import math
MOD = 10**9 + 7
n = 200000
prime_exps = [5, 12, 7, 3]  # exponents of primes in n!
ans = 1
for e in prime_exps:
    ans = ans * (math.factorial(n + e - 1) // (math.factorial(e) * math.factorial(n - 1)))
print(ans % MOD)
```
After:
```py
MOD = 10**9 + 7
n = 200000
prime_exps = [5, 12, 7, 3]
ans = 1
for e in prime_exps:  # multiplicative C(n+e-1, e) with early modulus
    num = den = 1
    for i in range(1, e + 1):
        num = num * (n - 1 + i) % MOD
        den = den * i % MOD
    ans = ans * (num * pow(den, MOD - 2, MOD) % MOD) % MOD
print(ans)
```
