---
skill_id: O002
type: operator
language: python
family: graph
name: Bfs Constructive Kth Digit Sequence
description: Reusable optimization pattern distilled from weighted traces.
tags:
- bfs
- implicit_graph
- sequence_generation
- kth_element
- constructive_enumeration
- digit_constraints
- recursion_to_iteration
- sorting_avoidance
- dp_elimination
- constant_factor_optimization
triggers:
- You need the k-th smallest value in a monotone sequence defined by local digit constraints (e.g., constraint only depends
  on neighboring digits).
- Existing code precomputes all valid numbers up to a fixed digit length or numeric bound, stores them, and then sorts to
  pick the k-th element.
- A digit-DP counting function count(x) is used inside a binary search on x, but there is only one (or few) k-th queries and
  k is moderate.
- Recursive DFS explores all digit strings up to some max length, often with sys.setrecursionlimit raised, regardless of the
  requested k.
- A heap-based approach maintains a global priority queue of candidates with a visited set and possibly a final sort or post-processing.
- Brute-force next-number search increments integers and repeatedly checks the digit property via string conversion and digit
  scanning.
- You see magic numeric cutoffs (e.g., fixed max value or max digit length) and significant over-generation unrelated to k.
---

## When to use
- You need the k-th smallest value in a monotone sequence defined by local digit constraints (e.g., constraint only depends on neighboring digits).
- Existing code precomputes all valid numbers up to a fixed digit length or numeric bound, stores them, and then sorts to pick the k-th element.
- A digit-DP counting function count(x) is used inside a binary search on x, but there is only one (or few) k-th queries and k is moderate.
- Recursive DFS explores all digit strings up to some max length, often with sys.setrecursionlimit raised, regardless of the requested k.
- A heap-based approach maintains a global priority queue of candidates with a visited set and possibly a final sort or post-processing.
- Brute-force next-number search increments integers and repeatedly checks the digit property via string conversion and digit scanning.
- You see magic numeric cutoffs (e.g., fixed max value or max digit length) and significant over-generation unrelated to k.

## Steps
1. {'step': 'Model_state_space', 'detail': 'Recognize that valid numbers form nodes in an implicit graph: state = current integer value; edges append a digit so that the local digit constraint (e.g., abs(new_digit - last_digit) <= 1) holds.'}
2. {'step': 'Choose_seeds', 'detail': 'Identify all minimal valid numbers as seeds (typically all 1-digit valid numbers without leading zeros). Initialize a queue (collections.deque) or list with these seeds in increasing order.'}
3. {'step': 'BFS_enumeration', 'detail': 'Run a loop for i in range(k-1): pop the leftmost element cur from the queue, generate up to a small constant number of children using the transition rule, and append them to the right of the queue in digit order. The
4. {'step': 'Return_answer', 'detail': 'After k-1 pops, the next element popped from the queue is exactly the k-th smallest valid number. Output this value directly without any global sort or further processing.'}
5. {'step': 'Memory_control', 'detail': 'Do not store all generated numbers separately; the queue itself is the working set. Optionally cap generation by stopping as soon as you reach k pops so the queue size stays O(k).'}
6. {'step': 'Implementation_details', 'detail': 'Use integer arithmetic (// and %) for digit operations instead of converting to and from strings. Use an iterative loop (while/for) rather than recursion to avoid recursion overhead and stack limits.'}

## Complexity
- Time: Typically O(k * B) where k is the requested index and B is the constant branching factor (often ≤ 3). For a single query with k up to ~1e5, this is effectively O(k). It replaces O(M log M) full-enumeration-and-sort, or O(D * log U * A)
- Space: O(k) for the queue and a few scalars, since only the first O(k) valid numbers are ever in memory. This replaces O(M) storage of all candidates or repeated allocation of DP tables.

## Pitfalls
- {'Falling_back_to_sort': 'Generating many or all valid numbers (via DFS, sets, or frontier layers) and then calling sorted( ) just to index the k-th value, instead of leveraging BFS order to get the k-th directly.'}
- {'Overgeneration_with_fixed_bounds': 'Driving enumeration by a fixed maximum digit length or numeric cutoff rather than by k, leading to unnecessary work when k is small relative to the search space.'}
- {'Unnecessary_digit_checks': 'Re-validating each generated child via a helper that scans all digits (often using str(num)) even though the transition rule already guarantees the constraint inductively.'}
- {'Using_heaps_where_queue_suffices': 'Maintaining a min-heap and visited set for what is essentially a simple layered generation where FIFO order already matches numeric order, paying O(log n) per operation plus
- {'Deep_recursion': 'Keeping a recursive DFS with increased recursionlimit for a shallow but wide search tree, causing avoidable Python call overhead and potential stack issues.'}
- {'Set_frontiers_plus_final_sort': 'Using sets for frontiers and a global set of all generated numbers, then sorting the final set, instead of a single queue with at most O(k) active elements.'}
- {'String_based_digit_manipulation': 'Reliance on repeated int↔str conversion and slicing to access digits in the inner loop, causing large constant factors; this is unnecessary when simple arithmetic suffices.'}
- {'Ignoring_k_scale': 'Preferring asymptotically attractive digit-DP + binary search (O(log U)) even when k is small/moderate and the DP involves heavy nested loops and repeated allocation, making it slower in practice

## When not to use
- {'When_you_need_many_random_accesses': 'If you must answer many queries for arbitrary ranks or ranges (e.g., many different k, or count of valid numbers up to x across many x), a precomputed DP or counting structure
- {'When_constraints_are_nonlocal': 'If the validity condition depends on global properties of the number (e.g., sum of digits modulo something, presence/absence of patterns over the whole string) that are not easily
- {'When_numeric_order_differs_from_BFS_order': 'If the natural graph or automaton order does not align with numeric or lexicographic order, you may need a priority queue or selection algorithm; a plain FIFO BFS will not
- {'Under_strict_memory_limits': 'If memory limits are extremely tight and k is very large, even O(k) queue storage may be problematic, in which case a streaming or on-the-fly next-number generator with O(1) extra space

## Minimal example
Before:
```py
# O002 focus: bfs
mat = csr_matrix((data,(u,v)), shape=(n,n))
comp = connected_components(mat, directed=True)
ans = solve_from_components(comp)
```
After:
```py
# optimized for bfs
adj = [[] for _ in range(n)]
for u, v in edges: adj[u].append(v)
ans = topo_dp(adj)
```
