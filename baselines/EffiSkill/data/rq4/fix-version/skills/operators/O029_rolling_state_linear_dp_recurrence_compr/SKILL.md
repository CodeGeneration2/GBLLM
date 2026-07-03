---
skill_id: O029
type: operator
language: python
family: dp
name: Rolling State Linear DP & Recurrence Compression
description: Transform linear, time-stepped dynamic programs or simple recurrences that only depend on a fixed number of previous
  states into memory-light, constant-factor-optimized implementations. The operator replaces full history tables and bulky
  input storage with rolling O(1) state, streaming input, and simplified transition structure. For DP with small fixed state
  dimension, it unrolls tiny inner loops, removes redundant
tags:
- dynamic_programming
- 1d_dp
- rolling_dp
- state_compression
- sequence_dp
- linear_recurrence
- space_optimization
- constant_factor_optimization
- streaming_io
- forward_dp
triggers:
- DP state indexed by a time/position i where transitions at step i depend only on states from step i-1 (or a fixed-size window
  of previous steps).
- DP table shaped as N x K where K is a very small fixed constant (e.g., 2–5), and only the last row is needed for the final
  answer.
- Array-based recurrence dp[i] = f(dp[i-1], dp[i-2], , dp[i-c]) with c being a small constant, while only dp[n] is ever used.
- Code stores both a full input matrix and a full DP table, but accesses input and DP strictly in increasing time order without
  random access.
- Inner loop iterates over a tiny fixed state set with if/elif branches per state (e.g., if s == 0 / 1 / 2), updating the
  same target cells multiple times.
- DP or recurrence implemented with large preallocated arrays (possibly with fixed caps like 1_000_000) when only O(1) recent
  values are required.
- 1D DP over indices 0..L where the goal concerns reaching at least some threshold H, yet transitions also treat indices ≥
  H as full base states.
- Hot inner loops performing redundant guard computations (e.g., max(0, i - a[j])) or transitions from unreachable states
  with INF-like sentinels.
---

## When to use
- DP state indexed by a time/position i where transitions at step i depend only on states from step i-1 (or a fixed-size window of previous steps).
- DP table shaped as N x K where K is a very small fixed constant (e.g., 2–5), and only the last row is needed for the final answer.
- Array-based recurrence dp[i] = f(dp[i-1], dp[i-2], , dp[i-c]) with c being a small constant, while only dp[n] is ever used.
- Code stores both a full input matrix and a full DP table, but accesses input and DP strictly in increasing time order without random access.
- Inner loop iterates over a tiny fixed state set with if/elif branches per state (e.g., if s == 0 / 1 / 2), updating the same target cells multiple times.
- DP or recurrence implemented with large preallocated arrays (possibly with fixed caps like 1_000_000) when only O(1) recent values are required.
- 1D DP over indices 0..L where the goal concerns reaching at least some threshold H, yet transitions also treat indices ≥ H as full base states.
- Hot inner loops performing redundant guard computations (e.g., max(0, i - a[j])) or transitions from unreachable states with INF-like sentinels.

## Steps
1. Identify local-time dependency: Verify that the DP or recurrence at step i depends only on a fixed set of previous steps (e.g., i-1, i-2, or previous row in a small K-state DP).
2. Identify local-time dependency: Confirm that the final result uses only the last state (or a simple aggregation over the last step), not the full history.
3. Compress DP state into rolling variables: A small tuple of scalars for simple recurrences (e.g., two values for second-order recurrences).
4. Compress DP state into rolling variables: Update these in each iteration via tuple assignment so all transitions use the previous state consistently.
5. Eliminate full-history and extra input storage: Remove arrays that store all input time steps if input is consumed strictly in order.
6. Eliminate full-history and extra input storage: Read each step’s data (e.g., a small tuple of values) and immediately apply the DP/recurrence update (streaming DP).
7. Eliminate full-history and extra input storage: If some DP history is still needed (e.g., for reconstruction), keep only what is logically required instead of duplicating input and DP.
8. Unroll tiny inner loops and remove redundant branching: For small fixed K, replace generic nested loops and if state==k branches with direct, explicit formulas for each state.

## Complexity
- Time: (pattern dependent)
- Space: Reduces DP state space from O(N * K) or O(N) to O(K) or O(1) when only a fixed number of previous states is needed. Eliminates full input storage by streaming when possible, cutting peak memory from O(N * K + input) to O(K) plus minimal

## Pitfalls
- Incorrect rolling updates: Updating some state variables in-place and then using them to compute other states in the same step, effectively mixing old and new values.
- Incorrect rolling updates: Fix by always computing next-step states from a snapshot of the previous state (tuple assignment or separate prev array).
- Accidental loss of necessary history: Removing the DP table entirely when later logic (e.g., path reconstruction or multiple queries) actually depends on intermediate states.
- Accidental loss of necessary history: Ensure that only the final value (or a simple aggregate) is required before discarding history.
- Breaking constraints when unrolling transitions: When replacing branches with formulas, carefully encode constraints via index selection or masks.
- Off-by-one errors on indices and thresholds: Changing DP index ranges (e.g., using 0..H-1 instead of 0..H+maxA) can easily shift base or terminal conditions.
- Off-by-one errors on indices and thresholds: Always re-derive and test base cases and termination conditions after reducing ranges.
- Over-aggressive state pruning: Assuming certain DP states are unreachable and skipping them without verifying reachability under all input constraints.

## When not to use
- When future logic requires full DP history: If you must reconstruct the actual sequence of choices, answer multiple offline queries over different prefixes, or inspect intermediate states, removing the DP table
- When transitions depend on long-range or non-local history: If dp[i] depends on many previous positions or on aggregate properties of the full prefix, rolling over a constant number of states is not sufficient.
- When the state dimension is large or data-dependent: If the number of states K is not a small fixed constant (e.g., K scales with N or input parameters), manual unrolling and scalarization may hurt clarity, and memory
- When correctness of pruning is uncertain: If you cannot rigorously justify that high-index DP states or INF-marked states never need to be used as bases, do not restrict the iteration range or skip their transitions.
- When input/output costs dominate and cannot be improved: If runtime is already dominated by network I/O, file system latency, or external system calls, micro-optimizing DP state representation will have limited effect.

## Minimal example
Before:
```py
# O029 focus: rolling
for i in range(n):
    dp[i] = max(dp[j] + 1 for j in range(max(0, i-w), i))
```
After:
```py
# optimized for rolling
for i in range(n):
    dp[i] = seg.query(lo(i), hi(i)) + 1
    seg.update(pos(i), dp[i])
```
