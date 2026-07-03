---
skill_id: O020
type: operator
language: python
family: state_compression
name: Set Intersection or Frequency Array Refactor for Common Elements Queries
description: Reframe membership-heavy, nested-loop logic that checks which values satisfy all group-wise conditions into either
  (a) a running intersection of hash sets, or (b) a single global frequency/count array over a small value range. This removes
  repeated list scans and list-based membership/removal, collapsing O(N*M^2) or O(N^2*K^2) patterns into O(total_input) time
  while often reducing memory. Use it whenever the task
tags:
- set_intersection
- frequency_count
- hash_set
- nested_loops_elimination
- data_structure_upgrade
- algorithmic_refactor
- membership_testing
- input_streaming
- python_optimization
triggers:
- Triple or double nested loops over groups and item IDs where the innermost body does `x in list` or `list.remove(x)`.
- 'Code conceptually answering "which values appear in every group" implemented as: for v in range(1..M): for each group:
  check membership of v by scanning a list.'
- A shrinking list of candidates frequently tested with `x in candidates` and mutated inside loops instead of using a set.
- Counting how many lists contain a value by rescanning all lists for each value rather than aggregating counts once.
- Use of NumPy or Counter.most_common() solely to count common items, with patterns like `np.append` in a loop or `np.sum(arr
  == v)` for every v.
- Full 2D tables (N×M) or list-of-lists of preferences stored when only final per-value consensus (intersection) is needed.
- Special-case branches such as `if N == 1` for the trivial single-group case, indicating the general logic is not expressed
  as a direct intersection.
- Post-processing lists with `len(set(result_list))` to deduplicate items that could have been unique if a set/intersection
  had been used earlier.
---

## When to use
- Triple or double nested loops over groups and item IDs where the innermost body does `x in list` or `list.remove(x)`.
- Code conceptually answering "which values appear in every group" implemented as: for v in range(1..M): for each group: check membership of v by scanning a list.
- A shrinking list of candidates frequently tested with `x in candidates` and mutated inside loops instead of using a set.
- Counting how many lists contain a value by rescanning all lists for each value rather than aggregating counts once.
- Use of NumPy or Counter.most_common() solely to count common items, with patterns like `np.append` in a loop or `np.sum(arr == v)` for every v.
- Full 2D tables (N×M) or list-of-lists of preferences stored when only final per-value consensus (intersection) is needed.
- Special-case branches such as `if N == 1` for the trivial single-group case, indicating the general logic is not expressed as a direct intersection.
- Post-processing lists with `len(set(result_list))` to deduplicate items that could have been unique if a set/intersection had been used earlier.

## Steps
1. Identify the logical goal: usually either (a) values that appear in every group, or (b) for each value, the number of groups it appears in.
2. Check the value domain: if item IDs are small integers in a known range [1..M], consider a frequency array; otherwise, use sets.
3. Set-intersection strategy (general IDs): Initialize a candidate set, typically `candidates = set(range(1, M+1))` if M is known, or from the first group’s items.
4. Set-intersection strategy (general IDs): For each group, parse its items and build a Python set of those items (`group_set`).
5. Set-intersection strategy (general IDs): Intersect iteratively: `candidates &= group_set`.
6. Set-intersection strategy (general IDs): At the end, `len(candidates)` is the count of items common to all groups; `candidates` itself is the intersection.
7. Frequency-array strategy (small integer domain): Allocate `cnt = [0] * (M+1)` (or length M) initialized to zero.
8. Frequency-array strategy (small integer domain): Stream input groups; for each item `a` in a group, increment `cnt[a]` (or `cnt[a-1]`). Avoid storing all groups if not needed.

## Complexity
- Time: (pattern dependent)
- Space: Reduces auxiliary space from storing all lists or dense N×M tables (O(N*M)) to O(M) for a single candidate set or frequency array, plus at most O(K_max) for a temporary per-group set. In worst-case fully dense inputs, set-based approaches

## Pitfalls
- Continuing to use lists for membership (`x in list`) and removal (`list.remove(x)`) in hot loops instead of converting to sets or using a count array.
- Building all per-group structures (list-of-lists or full DP tables) before any aggregation, missing the opportunity to stream input and intersect/count on the fly.
- Misusing NumPy: using `np.append` inside loops and repeated `np.sum(arr == v)` scans for each candidate v, causing quadratic behavior and heavy allocation overhead.
- Initializing the candidate set incorrectly, e.g., as empty set and then intersecting (which will always stay empty) instead of starting from the first group or full universe.
- Double-counting items when using frequency arrays by not deduplicating within a single group if the same item can appear multiple times in that group and semantics require per-group presence only.
- Mixing intersection and counting approaches (e.g., computing both a candidate set and a frequency array) without need, increasing complexity and memory.
- Forgetting that list.pop(0) and list.remove are O(L) operations; using them in inner loops negates gains from other optimizations.
- Relying on custom binary search or manual searching of sorted lists for each query instead of constructing sets when only membership/overlap counts are needed.

## When not to use
- When you need to preserve per-group structure, order, or multiplicity (e.g., multiset operations, sequence alignment) rather than just the set of values common to all groups.
- When the value domain is extremely large or sparse (e.g., arbitrary strings or huge integers) and you already maintain a minimal representation; then a frequency array over [1..M] may be impossible, and set
- When intersection must be computed repeatedly with many different subsets or under dynamic updates; in that case, more specialized indexing structures or incremental algorithms may be better than recomputing
- When the problem requires additional properties beyond commonality, such as positions, order of first appearance, or ties by frequency that require full ranking; then Counter with sorting or more complex data

## Minimal example
Before:
```py
kids_likes = [[1, 2, 3], [2, 3, 4], [2, 3]]
common = list(range(1, 5))  # food IDs 1..4
for food in common[:]:
    for likes in kids_likes:
        if food not in likes:  # repeated list scan
            common.remove(food)
            break
print(common)
```
After:
```py
kids_likes = [[1, 2, 3], [2, 3, 4], [2, 3]]
common_set = set(kids_likes[0])
for likes in kids_likes[1:]:
    common_set &= set(likes)  # fast hash-set intersection
common = sorted(common_set)
print(common)
```
