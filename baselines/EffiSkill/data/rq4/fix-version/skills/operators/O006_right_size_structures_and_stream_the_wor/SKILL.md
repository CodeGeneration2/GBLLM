---
skill_id: O006
type: operator
language: python
family: streaming
name: Right Size Structures and Stream the Work
description: Refactor linear-time counting and pair-counting code to remove large fixed buffers, redundant passes, and heavyweight
  structures. Use streaming prefix-style scans, boolean or frequency arrays sized to real input bounds, or hash maps over
  sparse keys. Keep the big-O the same but drop hidden O(K) terms, magic constants, and extra scans so runtime and memory
  scale with actual data instead of arbitrary limits.
tags:
- optimization
- counting
- prefix-sum
- frequency-map
- hash-map
- boolean-array
- sweep-line
- one-pass
- space-optimization
- constant-factor-optimization
triggers:
- Allocation of a large fixed-size list (e.g., 10**7, 400005, 86400, 100000) whose size is unrelated or only loosely related
  to the input size.
- A dedicated loop whose only job is to append zeros or initialize a large buffer before actual processing.
- Use of set(large_list) or Counter(large_list) over arrays that are mostly default values or padding.
- Loops that always iterate over a full fixed range (e.g., all seconds in a day, all 1..K buckets) even when only a small
  subset of indices carry information.
- Dense frequency arrays used with sparse or unbounded index expressions like i ± a[i].
- Construction of full prefix/suffix arrays when only a running aggregate (sum, min, max) is required.
- Using list.count inside a loop over possible values, or repeated `value in list` / list.remove inside nested loops.
- Post-processing passes that build sets or scan entire arrays just to count distinct roots, distinct IDs, or zero entries.
---

## When to use
- Allocation of a large fixed-size list (e.g., 10**7, 400005, 86400, 100000) whose size is unrelated or only loosely related to the input size.
- A dedicated loop whose only job is to append zeros or initialize a large buffer before actual processing.
- Use of set(large_list) or Counter(large_list) over arrays that are mostly default values or padding.
- Loops that always iterate over a full fixed range (e.g., all seconds in a day, all 1..K buckets) even when only a small subset of indices carry information.
- Dense frequency arrays used with sparse or unbounded index expressions like i ± a[i].
- Construction of full prefix/suffix arrays when only a running aggregate (sum, min, max) is required.
- Using list.count inside a loop over possible values, or repeated `value in list` / list.remove inside nested loops.
- Post-processing passes that build sets or scan entire arrays just to count distinct roots, distinct IDs, or zero entries.

## Steps
1. Right-size the data structure: If the domain is a small known range 1..N, prefer a list or boolean array of length N over a fixed oversized buffer or a set over a padded list.
2. Right-size the data structure: If keys are sparse (e.g., i ± a[i] with large or unknown bounds), replace dense arrays with dict/defaultdict/Counter keyed by the transformed value.
3. Remove artificial global constants: derive array sizes from actual constraints (n, max_value, etc.), or eliminate the array entirely in favor of hash maps when bounds are loose.
4. Stream the computation in a single pass where possible: For subarray or prefix-based counting, maintain a running prefix sum (or other aggregate) and a frequency map, updating the answer incrementally.
5. Stream the computation in a single pass where possible: For segment-based or greedy scans, keep only the minimal state (running min/max, current segment sum, overlap count) instead of full prefix/suffix arrays.
6. Replace heavy distinct-counting patterns: For union–find components, maintain a component counter updated in union instead of building a set of roots at the end.
7. Optimize pair-counting: Re-express the pair condition as equality of a simple key (e.g., prefix sum, i + a[i], i − a[i]).

## Complexity
- Time: (pattern dependent)
- Space: Reduces from O(K) auxiliary space (K = large fixed bound) or multiple O(n)-sized arrays to O(n) or O(k) where k is the number of distinct keys actually used. In many cases auxiliary space drops to O(1) plus a hash map or small fixed arrays.

## Pitfalls
- Treating large fixed constants as harmless because they don’t change big-O, while they actually dominate runtime or memory (e.g., always paying O(86400) or O(10**7) per test case).
- Overusing dense arrays for sparse index spaces, leading to memory errors or unnecessary initialization cost.
- Blindly replacing arrays with hash maps in extremely tight inner loops where the index range is truly small and dense, which can be slower than a compact list.
- Relying on padding values (like zero) plus len(set( )) and hand-tuned formulas instead of a direct representation, which both hurts clarity and risks off-by-one or correctness bugs.
- Using set or Counter on large sequences when only simple integer frequencies or boolean flags are needed.
- Keeping multi-pass designs (e.g., build full prefix/suffix arrays, then scan again) when a single streaming pass can accumulate all needed information.
- Implementing combinations via factorial or sum(range(k)) for k up to O(n), causing unnecessary quadratic or superlinear work.
- Migrating to list-based membership operations (x in list, list.remove(x)) as a replacement for set or boolean markers, inadvertently turning linear tasks into quadratic ones.

## When not to use
- When the large fixed range is genuinely small and dense (e.g., digits 0–9, small alphabet) and a simple fixed-size array is already optimal.
- When problem constraints guarantee that the fixed buffer is tiny relative to n and the implementation is already comfortably within time and memory limits.
- When hash map overhead would dominate because keys are dense small integers and performance is extremely sensitive to constant factors (then a tight list-based frequency array is preferable).
- When the logic fundamentally needs random access to full prefix/suffix arrays at many positions; in such cases, one-pass streaming with O(1) state is not equivalent.
- When a supposedly "better" algorithm introduces more complex data structures (e.g., sets, dicts) on very small inputs where the simpler original version is clearer and fast enough.

## Minimal example
Before:
```py
def count_zero_subarrays(arr):
    MAXN = 10**7
    pref = [0] * MAXN  # huge, input-independent buffer
    n = len(arr)
    for i in range(n):
        pref[i + 1] = pref[i] + arr[i]
    return sum(1 for i in range(n + 1) for j in range(i + 1, n + 1) if pref[j] - pref[i] == 0)
```
After:
```py
from collections import defaultdict

def count_zero_subarrays(arr):
    freq = defaultdict(int)
    s = ans = 0
    for x in arr:  # one streaming pass, structure sized to data
        freq[s] += 1; s += x; ans += freq[s]
    return ans
```
