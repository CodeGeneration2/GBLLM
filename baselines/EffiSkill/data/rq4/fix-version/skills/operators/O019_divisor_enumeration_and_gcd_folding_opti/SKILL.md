---
skill_id: O019
type: operator
language: python
family: coprimality
name: Divisor Enumeration and GCD Folding Optimization
description: 'Recognize when a brute-force numeric loop is really a divisor or common-divisor problem, then replace linear
  or nested scans with: (1) sqrt-based divisor enumeration plus simple filtering, or (2) Euclidean GCD folding over an array.
  This converts O(V) or O(N·sqrt V) patterns into O(√V) or O(N log V), often by algebraically simplifying congruence conditions
  into direct divisibility and exploiting divisor pairing.'
tags:
- number_theory
- divisors
- gcd
- euclidean_algorithm
- optimization
- search_space_reduction
- array_aggregation
- sqrt_optimization
triggers:
- A for- or while-loop whose bound is proportional to a raw numeric input or a ratio of inputs (e.g., range(1, B // A + 1))
  performing only modulo or simple arithmetic inside.
- Conditions of the form (X - k * Y) % k == 0, or similar algebra, where the loop variable is also the modulus base, suggesting
  the condition can simplify to X % k == 0 (k divides X).
- Searching for 'largest k' or 'any k' satisfying divisibility plus an inequality (e.g., k divides M and k <= M // K), implemented
  as a brute-force scan over k instead of over divisors.
- Loops that enumerate all integers up to sqrt(N) and perform divisibility checks, especially if repeated scans or nested
  loops over array elements are present.
- Algorithms on arrays that repeatedly apply modulo, deduplicate via set, and sort in a loop until values converge to a single
  number, indicating an implicit GCD computation.
- Use of len(set( )) purely to deduplicate divisor-like candidates that were aggressively over-generated in previous loops.
- 'Performance symptoms: O(V) or O(N·sqrt V) designs timing out for large numeric inputs, where constraints suggest divisor
  enumeration or GCD is the intended solution.'
---

## When to use
- A for- or while-loop whose bound is proportional to a raw numeric input or a ratio of inputs (e.g., range(1, B // A + 1)) performing only modulo or simple arithmetic inside.
- Conditions of the form (X - k * Y) % k == 0, or similar algebra, where the loop variable is also the modulus base, suggesting the condition can simplify to X % k == 0 (k divides X).
- Searching for 'largest k' or 'any k' satisfying divisibility plus an inequality (e.g., k divides M and k <= M // K), implemented as a brute-force scan over k instead of over divisors.
- Loops that enumerate all integers up to sqrt(N) and perform divisibility checks, especially if repeated scans or nested loops over array elements are present.
- Algorithms on arrays that repeatedly apply modulo, deduplicate via set, and sort in a loop until values converge to a single number, indicating an implicit GCD computation.
- Use of len(set( )) purely to deduplicate divisor-like candidates that were aggressively over-generated in previous loops.
- Performance symptoms: O(V) or O(N·sqrt V) designs timing out for large numeric inputs, where constraints suggest divisor enumeration or GCD is the intended solution.

## Steps
1. Algebraically simplify congruence conditions: whenever you see expressions like (X - k * Y) % k == 0, rewrite them in terms of simple divisibility (typically X % k == 0) to expose that k must be a divisor of some number.
2. Reformulate the goal as a divisor problem: express the requirement as 'find a divisor d of M satisfying some inequality or bound', e.g., d divides M and d <= M // K or K * d <= M.
3. Replace linear scans over candidate integers with sqrt-based divisor enumeration: iterate i from 1 to floor(sqrt(M)), and when M % i == 0, treat both i and M // i as divisors.
4. Generate and manage divisors efficiently: avoid repeated trial division over the same numeric range; centralize divisor generation into a helper that handles perfect squares and small/large divisor pairing cleanly.
5. Apply constraints after enumeration: filter the divisor set by inequalities (such as d <= limit or K * d <= M) and select the maximum or otherwise best candidate, optionally using sorting and/or binary search if that simplifies logic.
6. For array problems that converge via repeated modulo operations, recognize the task as computing the GCD of all elements, and replace complex loops with a standard Euclidean GCD fold (running_gcd = gcd(running_gcd, value) over the array).
7. Eliminate repeated global reorganization: remove patterns like while len(a) > 1: a = sorted(list(set(a))); a[i] %= a[0]; in favor of scalar GCD updates that do not require deduplication or sorting at each step.
8. Add cheap early exits: handle trivial cases such as exact divisibility (e.g., if B % A == 0, return B // A) or GCD == 1 early, to avoid unnecessary enumeration or folding.

## Complexity
- Time: Typically improves from O(V) or O(V / K) brute-force scans and O(N·sqrt V) divisor-checking loops to O(√V) for single-number divisor problems and O(N log V) for array GCD aggregation. Additional log factors (e.g., O(D log D)) may appear
- Space: Usually O(1) to O(√V) extra space. Pure GCD folding uses O(1) extra space; divisor-based methods that store all divisors use O(D) space, where D is the number of divisors, which is sublinear in V.

## Pitfalls
- Keeping the original O(V) loop and merely adding divisor helpers, causing extra overhead without actually changing the algorithmic bound.
- Forgetting the algebraic simplification and continuing to compute expressions like (B - k * A) % k inside large loops instead of reducing to B % k.
- Enumerating all integers up to sqrt(N) multiple times instead of reusing a precomputed divisor list, leading to redundant work and repeated factorization.
- Building full lists of divisors and then sorting them when only a running maximum is required, adding unnecessary O(D log D) overhead and O(D) memory usage.
- Implementing GCD-related logic via divisor enumeration per candidate and full-array checks, instead of using the Euclidean algorithm to compute GCD in O(log V) per combination.
- Ignoring worst-case patterns where an input parameter becomes very small (e.g., A = 1), turning a loop of length B // A into an infeasible O(B) iteration count.
- Mishandling perfect squares in divisor enumeration, either missing the sqrt(N) divisor or double-counting it, which can cause off-by-one errors or unnecessary deduplication.
- Switching to divisor enumeration but scanning divisors of a larger number when the constraint actually depends on a smaller or bounded one, losing potential constant-factor gains.

## When not to use
- When input sizes are small enough that a simple O(V) or O(N·sqrt V) solution is already comfortably within time limits and clarity is more important than optimization.
- When the condition in the loop does not reduce to simple divisibility or GCD properties, so divisor enumeration or Euclidean GCD does not capture the full logic.
- When the result requires enumerating or counting all divisors explicitly and constraints are tight enough that the O(√V) bound alone is insufficient (e.g., extremely large V with many test cases).
- When the algorithm is dominated by non-numeric overhead (I/O, complex data structures) rather than arithmetic loops; optimizing divisors or GCD in that case will not materially change performance.
- When using languages or environments where recursion depth or integer-division cost is prohibitive and a different numeric strategy (e.g., precomputed primes or specialized libraries) is more appropriate.

## Minimal example
Before:
```py
def best_box_size(total_candies, max_boxes):
    best = 1
    # brute-force over all possible box sizes
    for box_size in range(1, total_candies + 1):
        if total_candies % box_size == 0 and total_candies // box_size <= max_boxes:
            best = box_size
    return best
```
After:
```py
def best_box_size(total_candies, max_boxes):
    best = 1
    limit = int(total_candies ** 0.5)
    for d in range(1, limit + 1):
        if total_candies % d == 0:
            for box_size in (d, total_candies // d):
                if total_candies // box_size <= max_boxes and box_size > best:
                    best = box_size
    return best
```
