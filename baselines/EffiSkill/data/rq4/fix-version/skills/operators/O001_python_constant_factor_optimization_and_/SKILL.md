---
skill_id: O001
type: operator
language: python
family: constant_factor
name: Python Constant Factor Optimization and Lightweight Data Structure Refactor
description: A reusable optimization skill for taking an already-correct Python solution and making it fast enough for large
  inputs by aggressively reducing constant factors and replacing heavy abstractions with lightweight, specialized implementations.
  This includes (1) stripping template bloat (unused imports, helpers, and wrappers), (2) tightening hot loops and I/O, and
  (3) swapping overengineered data structures
tags:
- performance-optimization
- constant-factor-reduction
- code-cleanup
- data-structure-design
- python-specific
- segment-tree
- disjoint-set-union
- prefix-sums
- heap-optimization
- bfs-queue
triggers:
- Solution is already using an ostensibly optimal big-O algorithm but still times out or runs near limits for n up to ~1e5–3e5.
- 'Presence of a large competitive-programming template: many imports, helpers, and global constants that are unused or used
  only once.'
- Use of slow or heavy I/O patterns, such as eval(input()), many thin input wrappers, or per-line reads inside large loops.
- Data structure implemented via classes and methods (segment trees, Fenwick trees, heaps, DSU, BFS queues) with heavy attribute
  lookups and wrapper methods in the hot path.
- Range operations implemented with per-element loops over intervals (e.g., repeated point updates) despite constraints that
  demand O(log n) range updates/queries.
- Static query patterns (e.g., offline range sums or counts) solved with dynamic, general-purpose trees instead of prefix
  sums or DP tables.
- Use of thread-safe queues (queue.Queue) or other multi-purpose containers in purely single-threaded BFS or sequence generation.
- Inner loops showing repeated list allocations (slicing, reversed(list(range( ))), building temporary arrays) or redundant
  full-array passes.
---

## When to use
- Solution is already using an ostensibly optimal big-O algorithm but still times out or runs near limits for n up to ~1e5–3e5.
- Presence of a large competitive-programming template: many imports, helpers, and global constants that are unused or used only once.
- Use of slow or heavy I/O patterns, such as eval(input()), many thin input wrappers, or per-line reads inside large loops.
- Data structure implemented via classes and methods (segment trees, Fenwick trees, heaps, DSU, BFS queues) with heavy attribute lookups and wrapper methods in the hot path.
- Range operations implemented with per-element loops over intervals (e.g., repeated point updates) despite constraints that demand O(log n) range updates/queries.
- Static query patterns (e.g., offline range sums or counts) solved with dynamic, general-purpose trees instead of prefix sums or DP tables.
- Use of thread-safe queues (queue.Queue) or other multi-purpose containers in purely single-threaded BFS or sequence generation.
- Inner loops showing repeated list allocations (slicing, reversed(list(range( ))), building temporary arrays) or redundant full-array passes.

## Steps
1. Strip template and I/O overhead
2. - Remove all unused imports (especially heavy ones like numpy, decimal, fractions, itertools, collections where not needed).
3. - Delete unused helper functions, global constants, and direction vectors; keep only code that participates in the solution path.
4. - Replace eval(input()) with safe, direct parsing (int(input()), map(int, input().split()), or a single bulk read via open(0).read().split()).
5. - Collapse tiny I/O wrapper functions into a single fast reader (e.g., alias sys.stdin.readline) and call it directly in the main logic.
6. Tighten hot loops and arithmetic
7. - Inline trivial helpers used in inner loops (e.g., wrappers around max, min, abs, or simple arithmetic) to avoid extra function calls.
8. - Eliminate redundant passes over arrays: compute required aggregates (prefix/suffix sums, streak lengths, counts) in a single linear scan whenever possible.

## Complexity
- Time: Typically preserves the original asymptotic complexity (e.g., O(n), O(n log n)), while significantly reducing constant factors. In some applications, also improves asymptotics: e.g., upgrading range operations from O(range_length * log n)
- Space: Generally keeps the same asymptotic space as the original solution while reducing unnecessary allocations and temporary structures. In some refactors, space grows modestly (e.g., adding a DP array of size n for clarity) or changes form

## Pitfalls
- Over-optimizing without measurement: spending time micro-optimizing non-critical paths while leaving algorithmic bottlenecks (e.g., range operations implemented as repeated point updates) unchanged.
- Regressing asymptotics: replacing a linear or O(n log n) streaming solution with a naive sort or brute-force for convenience in cases where n can be large enough to make the change harmful.
- Incorrect range-tree redesign: introducing lazy segment trees without carefully handling push-down/pull-up logic, leading to subtle bugs on overlapping range updates or queries.
- Using heavy libraries unnecessarily: introducing vectorized tools (e.g., numerical libraries) for small n where import and conversion costs outweigh benefits, or where such libraries are not allowed.
- Neglecting numerical semantics: changing from float-based to integer-based arithmetic (e.g., ceil, rounding) without verifying that the integer formulation matches the required rounding rules.
- Breaking stability or ordering guarantees: simplifying sequence-generation BFS or greedy logic with new data structures but inadvertently changing the order in which items are produced.
- Memory blowups from static tables: replacing dynamic segment trees with prefix-sum grids or 2D arrays without confirming that O(n^2) or similar memory usage fits within limits.
- Unsafe or slow input parsing: leaving eval(input()) or complex input wrappers in place under the assumption that they are not a bottleneck when they are called millions of times.

## When not to use
- When the current solution already meets time and memory limits comfortably; further micro-optimization may reduce clarity without practical benefit.
- When constraints are very small (e.g., n ≤ 10^3) and algorithmic simplicity is more valuable than shaving constant factors.
- When using heavy external libraries solely for speed in environments where they are disallowed or where import time dominates the overall runtime.
- When an asymptotic improvement is clearly necessary (e.g., naive O(n^2) where n can be 2e5); in such cases, prioritize redesigning the algorithm over constant-factor tweaks.
- When memory is very tight and proposed replacements rely on dense 2D arrays or large prefix tables that risk exceeding limits.
- When correctness is delicate (complex game theory, intricate invariants) and the optimization would significantly obfuscate the logic; in such cases, first consider a clearer asymptotically efficient algorithm, then

## Minimal example
Before:
```py
def count_hits(data, queries):
    hits = []
    for q in queries:
        hits.append(sum(1 for x in data if x == q))
    return hits
```
After:
```py
def count_hits(data, queries):
    freq = {}
    for x in data:  # single pass, lightweight dict
        freq[x] = freq.get(x, 0) + 1
    return [freq.get(q, 0) for q in queries]
```
