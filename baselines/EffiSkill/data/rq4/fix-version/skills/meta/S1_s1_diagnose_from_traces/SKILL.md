---
skill_id: S1
type: meta
language: python
name: S1 Diagnose from Traces
description: Systematically diagnose why a competitive-programming solution is slow using structured audit traces (constraints,
  dominant operations, complexity, and bottlenecks).
---

## S1 — Diagnose (from weighted traces)

This skill explains how to read `ProblemBrief` and `SlowAudit` traces and turn them into a clear diagnosis of *why* a solution is slow and *what type* of optimization is needed.

The goal is **not** to design the fix yet, but to:
- Classify the main performance failure (e.g., wrong asymptotic complexity vs constant factors).
- Pinpoint which operations dominate time/space.
- Infer what structural change is likely required.

We assume a trace structure similar to the examples:
- `ProblemBrief.constraints_guess`
- `ProblemBrief.problem_type_tags`
- `SlowAudit.dominant_ops`
- `SlowAudit.complexity`
- `SlowAudit.bottlenecks`
- `trace_weight`

---

### 1. Start from the problem model, not the implementation

1. **Read `constraints_guess` first.**
   - Identify key magnitudes: `n_max`, `q_max`, and notes like "N up to 1e5" or "N up to 1e12".
   - Infer the *target* complexity:
     - If `n_max ≈ 1e5` and run is single-testcase, typical targets are `O(n)`, `O(n log n)`.
     - If `N` or values are up to `1e12+`, operations must be ~`O(sqrt(N))` or better, never `O(N)`.

2. **Check `problem_type_tags` to know the natural algorithmic shape.** Examples from the traces:
   - `"array", "one-pass-processing"` → likely intended `O(n)` scanning.
   - `"all_pairs_shortest_paths", "graph"` with `N ≤ 2e3` → `O(N^2)` often acceptable; `O(N^3)` or `O(N^2 log N)` may be too slow.
   - `"number_theory", "prime-factorization"` with `M ≤ 1e9` → `O(sqrt(M))` factorization is common.
   - `"combinatorics", "binomial_coefficient"` → check whether a closed form exists instead of summing many combinations.

This step gives you a **mental contract**: which asymptotic orders are acceptable.

---

### 2. Inspect the stated complexity vs the constraint contract

Use `SlowAudit.complexity`:

- Extract:
  - `time` (e.g. `"O(n)"`, `"O(N^2 log N)"`, `"O(n^2)"`).
  - `space`.
  - `why` (narrative justification).

Then **compare with the contract** from step 1:

1. **If stated time is asymptotically too large**, you already know the main diagnosis is *algorithmic*:
   - Example: *All-pairs shortest paths on a simple path+one edge*
     - Constraint guess: `N ≤ 2000`.
     - Trace: `time: O(N^2 log N)` using Dijkstra from every vertex.
     - For `N=2000`, `N^2 ≈ 4e6`, but `N^2 log N` (with Python overhead) is risky; BFS or a direct combinatorial `O(N^2)` formula is expected.
   - Example: *Parity subset counting*
     - Constraint guess: `n ≤ 1e5`.
     - Trace: `O(n^2)` big-int factorial work.
     - This is drastically too slow; the intended solution is `O(n)` or `O(n log n)` with a parity argument.

2. **If stated time matches the contract asymptotically**, then focus on *constant-factor* or *implementation* issues:
   - Example: *subarray sum zero count*
     - `n_max ≈ 2e5`.
     - Trace: `O(n)` time and `O(n)` space.
     - Diagnosis: fine asymptotics, but overhead from suffix array + `Counter` + extra checks.

3. **If time is borderline (e.g., `O(N^2)` at `N ≈ 2e3`)**, also check:
   - Whether extra log factors are present (`log N`, `log MOD`).
   - Whether heavy constant factors (e.g., Python Dijkstra, big-int factorials) are used.

At this stage you should be able to answer: **Is the failure primarily a wrong big-O class, or primarily excessive constant factors?**

---

### 3. Use `dominant_ops` to localize the hotspot

`SlowAudit.dominant_ops` lists the main operations that consume time.

For each entry, classify the *type*:

1. **Global loops over large ranges**
   - `"BFS from every starting node"` → pattern: *outer loop over N, inner BFS over N* → `O(N^2)`.
   - `"Two separate loops up to int(sqrt(N)) with modulo checks"` → pattern: *trial division / divisor enumeration* → `O(sqrt(N))`.
   - `"Linear search over all possible counts ..."` → pattern: *brute-force search* where algebraic solution exists.

2. **Nested computation-heavy loops**
   - `"Repeated computation of combinations via factorial inside loops"` → `O(n^2)` heavy math.
   - `"Precomputation of factorials, modular inverses up to O(N)"` → one-off `O(N)` precomputation.

3. **I/O and data-structure overhead**
   - `"Allocation and fill of suffix-sum array"`.
   - `"Queue operations in BFS"`.
   - `"Heap operations for priority queue"`.

Then decide which item is truly dominant by combining:
- Its order (`O(n)`, `O(N^2)`, `O(sqrt(M))`).
- How big the iterated range can get under the constraints.
- The cost per step (simple arithmetic vs big-int factorial vs heap operations).

Examples:

- In the subset counting trace, the real hotspot is not just the outer `O(n)` loops but specifically **big integer factorials** called `O(n^2)` times.
- In the knight move path-counting trace, the hotspot is **linear search + per-step modular exponentiation**; the complexity description confirms this as `O(X + K * log MOD)`.

---

### 4. Read `bottlenecks` to classify the failure mode

`SlowAudit.bottlenecks` provides labeled insights like `"algorithmic"`, `"precomputation"`, `"I/O_overhead"`, `"constant_factors"` etc. Use these to **categorize** the problem:

#### 4.1 Algorithmic mismatches

Look for labels and text indicating the algorithm itself is conceptually wrong for the problem scale:

- **Using a heavy general algorithm for a structured special case**
  - Dijkstra for an unweighted line+one-edge graph.
  - BFS from every node where a direct combinatorial formula exists.

- **Enumerating combinatorial objects instead of using identities/closed forms**
  - Explicitly summing `C(n, k)` for many k when `sum C(n,k) = 2^n` or parity arguments simplify.

- **Brute-force search for parameters that satisfy simple equations**
  - Linear search over counts of moves for `(1,2)` and `(2,1)` knight steps instead of solving the linear system directly.

Diagnosis label: *“Algorithmic mismatch / need different algorithmic strategy”*.

#### 4.2 Excessive asymptotics but same algorithm family

Sometimes the right "family" is used (e.g., factorization by trial division) but with avoidable extra loops:

- Separate `sqrt(N)` scans for `N` and `N-1` without reuse.
- Building a list of divisors then converting to `set` for deduplication, when checking duplicates up front would avoid extra hashes.

Diagnosis label: *“Redundant passes / avoidable full-range scans / better organization within same approach”*.

#### 4.3 Heavy precomputation vs actual query load

Typical patterns:

- Precomputing factorials up to `N+100` even when you only evaluate a handful of combinations.
- Sieve up to `A_max = 1e6` even if only a small subset of primes is needed.

Diagnosis label: *“Overkill precomputation; make precomputation adaptive or shrink its range”*.

#### 4.4 Constant-factor and implementation overhead

These do not change asymptotic complexity but affect runtime:

- Using `Counter` instead of `dict` in a hot loop.
- Rebuilding large arrays (`distance` arrays in BFS) inside inner loops.
- Python recursion for `gcd` instead of `math.gcd`.
- Unnecessary helper functions, lambdas, or `eval` for input.

Diagnosis label: *“Implementation-level optimization needed: reduce allocations, avoid dynamic dispatch, simplify operations per iteration”*.

---

### 5. Cross-check multiple traces (if weighted)

If you have several traces with `trace_weight`:

1. **Prioritize higher-weight traces**: they represent more common or important patterns.

2. **Look for recurring failure modes across traces**:
   - Repeated use of factorial-based combination loops in combinatorics problems.
   - Repeated `all-pairs shortest paths by N runs of SSSP` in similar graph problems.
   - Repeated global sieves or global factorial precomputation when queries are few.

3. From these repetitions, infer **general diagnostic heuristics**, e.g.:
   - *“If you see binomial coefficients inside loops with k up to n≈1e5, suspect an algorithmic simplification exists.”*
   - *“If `N` is small-ish (≤2e3) but `O(N^2 log N)` with heaps is used, check whether an `O(N^2)` combinatorial solution is expected.”*
   - *“If a sieve to 1e6 is used while input values seem small or sparse, question whether a full sieve is needed.”*

This helps you form **meta-diagnoses** that apply to new problems.

---

### 6. Summarize the diagnosis clearly

End by formulating a concise diagnosis along three axes:

1. **Asymptotic class vs required**
   - *“Current solution is `O(N^2 log N)` but the problem expects about `O(N^2)` or better for `N ≤ 2000`.”*
   - *“Current solution is `O(n^2)` big-int work; with `n ≤ 1e5` this is far beyond the intended `O(n)`.”*

2. **Dominant hotspot** (from `dominant_ops` and `bottlenecks`)
   - *“The time is dominated by repeated factorial computations inside loops.”*
   - *“The main overhead is multiple BFS/Dijkstra runs, one per node.”*
   - *“The main cost is an unconditional sieve/factorial precompute that dwarfs the tiny number of queries.”*

3. **Nature of the required fix** (without designing it yet)
   - *Algorithmic change*: replace enumeration with a formula, or Dijkstra/BFS-from-all with a combinatorial distance count.
   - *Structural change*: reuse computations, drop full-range scans, avoid repeated precomputation.
   - *Implementation tuning*: switch data structures, cut allocations, prefer built-ins.

Example summary pattern:

> For this problem (`N up to 1e5`, combinatorial counting), the current solution is `O(n^2)` due to repeated factorial-based combination calculations for many k. This is a fundamental algorithmic mismatch: the task only depends on simple parity properties, so a closed-form `O(n)` or `O(1)` solution is expected. The dominant operations are big-integer factorial computations and redundant recomputations of the same factorial values. A correct fix will require re-deriving the combinatorial formula and eliminating the explicit summation over k, not just micro-optimizing the factorial code.

This style of summary is the final output of **S1 — Diagnose**: a clear, justified statement of *why* the code is slow and *what class* of remedy is necessary.
