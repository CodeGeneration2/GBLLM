---
skill_id: O003
type: operator
language: python
family: combinatorics
name: Small k Binomial via Prime Exponents (No Global Factorials)
description: Reusable optimization pattern distilled from weighted traces.
tags:
- number_theory
- combinatorics
- prime_factorization
- binomial_coefficient
- modular_arithmetic
- precomputation_removal
- small_k_combinations
- time_optimization
- memory_optimization
triggers:
- Factorial and inverse-factorial arrays are precomputed up to N, N+offset, or a fixed MAX (e.g., 2e5) before using them for
  nCr, even though only O(log m) binomial coefficients are evaluated.
- Array sizes for factorial helpers are tied directly to a large input parameter (like N up to 1e5–1e9) or to a worst-case
  MAXN rather than to actual small k values.
- The only required combinations are of forms like C(n + e - 1, e) or C(n, e) where e comes from prime exponents and the sum
  of all e is provably O(log m).
- There is only one test case or very few test cases, so large factorial precomputation cannot be amortized across many queries.
- Runtime or profiling shows most time spent in factorial/inverse-factorial build loops or modular inverse-table construction,
  not in prime factorization or the core combinatorial logic.
- Memory usage is dominated by factorial/inverse arrays of size O(N) or O(MAXN), sometimes leading to memory limits, even
  though the logical math only depends on a few small exponents.
- Constraints allow parameters like N or M up to around 1e9, making any O(N) factorial precomputation or allocation clearly
  infeasible.
- Repeated use of pow(x, MOD-2, MOD) inside generic nCr for tiny k, when a direct integer multiplicative formula would suffice.
---

## When to use
- Factorial and inverse-factorial arrays are precomputed up to N, N+offset, or a fixed MAX (e.g., 2e5) before using them for nCr, even though only O(log m) binomial coefficients are evaluated.
- Array sizes for factorial helpers are tied directly to a large input parameter (like N up to 1e5–1e9) or to a worst-case MAXN rather than to actual small k values.
- The only required combinations are of forms like C(n + e - 1, e) or C(n, e) where e comes from prime exponents and the sum of all e is provably O(log m).
- There is only one test case or very few test cases, so large factorial precomputation cannot be amortized across many queries.
- Runtime or profiling shows most time spent in factorial/inverse-factorial build loops or modular inverse-table construction, not in prime factorization or the core combinatorial logic.
- Memory usage is dominated by factorial/inverse arrays of size O(N) or O(MAXN), sometimes leading to memory limits, even though the logical math only depends on a few small exponents.
- Constraints allow parameters like N or M up to around 1e9, making any O(N) factorial precomputation or allocation clearly infeasible.
- Repeated use of pow(x, MOD-2, MOD) inside generic nCr for tiny k, when a direct integer multiplicative formula would suffice.

## Steps
1. Confirm over-precomputation: Locate any global factorial, inverse, or inverse-factorial arrays (often sized N+offset or fixed MAXN) and associated loops that precompute them. Check that the number of actual nCr calls is only O(number_of_prime_factors), not
2. Design a small-k binomial implementation: Implement a direct multiplicative integer nCr function that runs in O(k) time where k is the smaller of r and n−r. Use the identity C(n, k) = Π_{i=1..k} (n + 1 − i) / i, or an equivalent formulation, and rely on
3. Integrate with prime factorization: Keep or lightly optimize the trial-division factorization logic (up to sqrt(m)), but ensure it returns exponents directly (e.g., dict{prime: exponent} or a Counter) so you can iterate over exponents without extra passes
4. Handle modular arithmetic efficiently: Decide between two safe strategies: (a) compute binomials as exact integers and apply the modulus only when multiplying into the global answer, or (b) compute the binomial under the modulus using multiplicative updates
5. Clean up and simplify code: Remove unused globals, helper templates, and I/O layers that were tied to the factorial-based approach. Ensure that the only arrays or maps that remain are for the factorization (e.g., a small dictionary or Counter) and scalar

## Complexity
- Time: Typical after applying this operator: O(√m + Σ e_i) or O(√m + Σ e_i · log MOD) where m is the factored integer and e_i are prime exponents (Σ e_i = O(log m)). The dominant term is usually the O(√m) trial-division factorization
- Space: O(#prime_factors(m)) or O(Σ e_i) extra space for storing factors/exponents and a few scalars. No large O(N) or O(MAXN) arrays for factorials or inverse factorials are required.

## Pitfalls
- Incorrect handling of integer division in nCr: When using multiplicative formulas like Π(n+1−i)/i, the division must be exact at each step or at the end (using integer arithmetic). Mixing modular arithmetic with plain
- Reintroducing large-N dependence through k: If k is incorrectly chosen as the larger side (e.g., using k instead of min(k, n−k)), loops may become unnecessarily long. In some formulations, misinterpreting parameters
- Not exploiting small exponent bounds: If the code does not rely on the mathematical fact that sum of exponents from factorization is O(log m), it might add extra layers (e.g., recursion, unnecessary data structures)
- Incorrect factorization loop bounds: Using a for-loop up to sqrt(initial_m) without updating the bound as m shrinks can cause extra trial divisions. More critically, failing to append the remaining m > 1 after the loop
- Assuming factorial-based logic for all inputs: The optimized operator is tailored to scenarios where the number of binomial calls and k values are tiny. Applying it blindly to a context with many large-k combinations
- Modulo timing and overflow assumptions: Delaying modulo too much can create very large intermediate integers; while Python can handle big ints, extreme inputs or nested products can lead to performance regressions. On
- Hard-coded precomputation remnants: Leaving constants like MAXN = 2*10**5 or arrays sized by MAXN in the code even though they are unused can confuse future maintainers and may accidentally be repurposed, reintroducing
- Edge cases: m = 1 or no prime factors: When m = 1, the product over primes is empty and the result is typically 1. Forgetting to short-circuit or correctly handle this can lead to running factorization and combination

## When not to use
- When the number of binomial coefficient evaluations is large and k is not small: If you need many nCr queries with moderate or large k (e.g., DP over combinations, convolution, or combinatorial tables), global
- When n and k are both large and comparable in scale: For general-purpose combinatorics with n, k up to around 2e5 and many queries, factorial-based O(1) nCr is preferable to O(k) multiplicative nCr per query.
- When factorization is not part of the structure: If your code does not naturally involve factoring an integer and extracting small exponents, forcing this pattern may add unnecessary O(√m) cost and complexity.
- When constraints are so small that precomputation overhead is negligible: If N and MAXN are tiny (e.g., ≤ 10^4) and memory is plentiful, keeping straightforward factorial-based nCr may be simpler and sufficiently fast
- When intermediate big integers become too large due to different parameter regimes: If n is huge and k is not very small, exact integer multiplication may create very large intermediate values and slow Python down. In

## Minimal example
Before:
```py
MOD = 10**9 + 7
N = 10**6
fact = [1]*(N+1)
for i in range(1, N+1):  # heavy precomputation despite few nCr uses
    fact[i] = fact[i-1]*i % MOD

def comb(n, k):
    return fact[n] * pow(fact[k]*fact[n-k] % MOD, MOD-2, MOD) % MOD

m = 10**12
n = 10**5
ans = 1
p = 2
while p*p <= m:  # factor m, use exponents as small k's
    if m % p == 0:
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        ans = ans * comb(n + e - 1, e) % MOD
    p += 1
if m > 1:
    ans = ans * comb(n, 1) % MOD
print(ans)
```
After:
```py
MOD = 10**9 + 7

def small_comb(n, k):  # O(k) multiplicative nCr, no global factorials
    if k < 0 or k > n: return 0
    k = min(k, n-k)
    num = den = 1
    for i in range(1, k+1):
        num *= n - k + i
        den *= i
    return num // den % MOD

m = 10**12
n = 10**5
ans = 1
p = 2
while p*p <= m:  # same factorization, but binomials are cheap now
    if m % p == 0:
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        ans = ans * small_comb(n + e - 1, e) % MOD
    p += 1
if m > 1:
    ans = ans * small_comb(n, 1) % MOD
print(ans)
```
