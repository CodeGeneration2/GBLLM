---
skill_id: S4
type: meta
language: python
name: S4 Judge Static Ranking
description: Rank candidate solutions by quality using static performance audits, weighing algorithmic complexity, constant
  factors, correctness risk, and problem context.
---

### What this skill is

S4 (Judge) is the ability to take one or more static audits (like the `SlowAudit`/`FastAudit` pairs and `DeltaSummary` objects above) and turn them into a **reasoned ranking of candidate solutions**.

You are not just describing differences; you are **judging** which version is better for the actual problem constraints, and why, including edge‑case risks. This is the skill used to say "Version B is the one we should ship / recommend".

Static ranking means:
- You rely on complexity analysis, structure of the code, and known patterns
- You do **not** rely on running or profiling the code

The weighted traces you’re given are the raw material that S4 consumes.

---

### Core behaviors

When you apply S4 to a set of candidate solutions and their audits, you should systematically:

1. **Identify the improvement type**
   - Use `DeltaSummary.delta_type` and complexity deltas to classify the change:
     - Pure constant‑factor tweaks (e.g. removing unused helpers, simpler I/O)
       - Example: mean + variance computation (trace 1) where both versions stay `O(n)` but Fast removes wrapper functions and unused lambdas.
     - Same‑asymptotic but structurally better (e.g. precompute divisors, reuse work)
       - Example: divisor processing for `N` and `N-1` (trace 2) where both remain `O(√N)` but Fast centralizes divisor generation and filters candidates.
     - True algorithmic improvements (lower big‑O)
       - Example: factorial‑based combinatorial sum vs parity argument (trace 3) going from super‑linear with big‑int factorials to `O(n)`.
       - Example: Dijkstra/BFS for all nodes vs closed‑form distances in a structured graph (trace 7 & 8) going from `O(N² log N)` to `O(N²)`.

   **Judging rule:** Prefer solutions with strictly better asymptotic complexity; otherwise, prefer those that reduce constant factors without harming clarity or correctness.

2. **Evaluate correctness and mathematical soundness**
   - Check if the faster code rests on a mathematically valid reduction:
     - Parity subset count (trace 3): confirm that “if any odd exists, exactly half of all subsets have odd sum” really holds under the problem’s parity semantics.
     - Line + one extra edge (traces 4, 7, 8): confirm the distance formula `min(|i-j|, |i-X|+1+|Y-j|, |i-Y|+1+|X-j|)` matches the original graph edges.
     - Lattice moves (trace 14): confirm the linear system for move counts and divisibility conditions are derived correctly.
   - Use `DeltaSummary.pitfalls` to spot risk:
     - Rounding semantics (trace 1): Python `round()` is banker's rounding; if the problem expects a different rule, both versions may be wrong.
     - Divisor enumeration (trace 2): verify perfect‑square handling and inclusion/exclusion of divisor `1`.
     - Combinatorics (multiple traces): verify `nCk` definitions, integer division, and modulus handling are correct.

   **Judging rule:** A slightly slower but obviously correct solution may outrank a fragile micro‑optimized one, especially when pitfalls indicate non‑obvious corner cases.

3. **Weigh asymptotics vs constraints**
   - Map complexity to typical constraints implied by the audits:
     - `O(n²)` in Python is often fine for `n ≲ 2e3` but not for `n ≈ 1e5`.
     - Heavy `math.factorial` or `pow(..., MOD-2, MOD)` inside loops (traces 3, 6, 11, 13, 14) will likely TLE for `n≈1e5` or large numbers of queries.
     - Full sieve to 1e6 (trace 15) is acceptable, but if the fast version can avoid it and still pass constraints, it is strictly better.

   Use the trace hints:
   - If `SlowAudit` notes calls to `math.factorial` or repeated modular inverses inside loops, treat this as a serious scalability red flag.
   - If `FastAudit` describes single‑pass `O(n)` or closed‑form formulas while Slow is quadratic or worse, strongly prefer Fast.

   **Judging rule:** Rank algorithms by their **effective runtime** under likely input sizes, not only by big‑O symbols. Use the narrative in the audits to infer realistic constraints.

4. **Assess constant‑factor and implementation overhead**
   Even with identical big‑O, implementations can differ a lot:
   - I/O and wrapper overhead (trace 1): many thin wrappers around `sys.stdin.readline` vs plain `input().split()`.
   - Data structure choices (trace 5): `collections.Counter` vs plain `dict` for prefix sums.
   - Graph machinery vs simple arithmetic (traces 4, 7, 8): custom graph classes + Dijkstra vs direct formulas.
   - Global precomputations (traces 6, 11, 13, 15): factorial tables or full sieves vs per‑query small loops or input‑driven sieving.

   **Judging rule:** If asymptotics match, prefer the design that:
   - Does fewer passes over data
   - Uses simpler data structures where appropriate
   - Avoids unnecessary global precomputation

5. **Factor in security and robustness smells**
   Some fast codes contain problematic patterns:
   - `eval(input())` (traces 1, 2, 6, 7, 8, 11, 13, 15): unsafe and often slower than `int(input())`.
   - Over‑reliance on Python recursion (trace 15) vs `math.gcd`.

   **Judging rule:** When performance is comparable, solutions that avoid unsafe or fragile constructs (like `eval`) should rank higher.

6. **Use trigger signals for deeper scrutiny**
   The traces explicitly give **triggers**; your job is to act on them:
   - `math.factorial` in loops or explicit summation over `C(n,k)` (trace 3) ⇒ strongly suspect inefficiency.
   - SSSP for each node in a structured graph (traces 4, 7, 8) ⇒ look for closed‑form distance formulas.
   - Full sieve / precomputation with few queries (traces 6, 11, 13, 15) ⇒ consider per‑query methods.
   - Double loops over `k` with parity constraints (trace 3) ⇒ search for parity arguments or identities.

   **Judging rule:** When signals are present, do not treat the slow code as viable at scale unless constraints are tiny or explicitly justify it.

7. **Synthesize a ranking, not just pairwise comparisons**
   When multiple candidate styles exist (e.g., different ways to compute `nCk` or factor numbers), S4 should:
   - Order them by:
     1. Correctness & conceptual clarity
     2. Asymptotic complexity
     3. Constant factors / implementation simplicity
   - Example ordering for `nCk` (from multiple traces):
     1. Direct multiplicative `nCk` with small `k`, integer division, then single modular reduction (traces 6, 11, 13, 14) – best when `k` is small, queries are few
     2. Precomputed factorial + inverse factorial tables – best when you have many large‑`k` queries
     3. Recomputing factorials per query or per `k` in loops – worst

   Similarly for graph problems in these traces:
   - Closed‑form distance formulas on structured graphs (line + one edge) ⇒ top choice
   - Single BFS/all‑pairs via combinatorial counting with `O(N²)` arithmetic ⇒ mid, acceptable if needed
   - Repeated Dijkstra/BFS from each node ⇒ bottom for such structured graphs

---

### How to apply S4 in practice

Given a new pair of audits `(SlowAudit, FastAudit, DeltaSummary)`:

1. **Read the `DeltaSummary` first**
   - Note `delta_type`: is this mostly constant‑factor or a genuine algorithmic change?
   - Note `complexity_delta`: understand asymptotic change and the narrative note.

2. **Scan `SlowAudit.bottlenecks` vs `FastAudit.core_idea`**
   - Identify which bottlenecks are eliminated and whether any new ones are introduced.
   - Example patterns:
     - Slow: repeated BFS/SSSP; Fast: `O(1)` distance formula per pair (traces 4, 7, 8).
     - Slow: factorial in loops; Fast: parity argument or direct small `nCk` (traces 3, 6, 11, 13, 14).

3. **Check `pitfalls` for both**
   - Confirm that the fast solution does not introduce new correctness risks worse than the slow solution’s.
   - If pitfalls are subtle or tied closely to the optimization, mentally discount some of the performance win for reliability.

4. **Draw a conclusion suitable for ranking**
   Concretely phrase a judgment such as:
   - "Fast is strictly better: it reduces time from `O(N² log N)` to `O(N²)` using a provably correct distance formula, with only straightforward arithmetic changes. It should be the recommended solution."
   - "Fast keeps `O(n)` but simplifies the code and cuts constant factors; both are viable, but Fast is preferred."
   - "Fast is asymptotically faster but relies on an assumption about rounding / parity that may not match the problem statement. Without confirming that assumption, I would treat Slow as the safer baseline."

5. **If multiple fast variants are available, order them**
   - Use the generic ordering rules above to output a consistent ranking.

---

### Mental checklist for S4

When ranking candidate solutions from static audits, quickly ask:

1. **Complexity**
   - Did the optimization reduce big‑O time or space?
   - If not, did it at least reduce passes, allocations, or per‑iteration cost?

2. **Correctness risks**
   - Does the fast version rely on identities or properties that must be re‑verified (parity, rounding, combinatorial identities)?
   - Are there off‑by‑one, indexing, or modular arithmetic pitfalls called out?

3. **Constraints alignment**
   - For plausible constraints, which version is actually safe from TLE / MLE?

4. **Implementation quality**
   - Is the code simpler or more complex after optimization?
   - Are unsafe constructs (`eval`, unnecessary recursion) removed or added?

5. **Final rank**
   - Given all of the above, which solution would you:
     - Recommend as the primary answer?
     - Keep as a backup/reference?
     - Discard as unsuitable under realistic constraints?

S4 is the skill of turning all this analysis into a crisp, defensible ranking – choosing the solution that balances performance, correctness, and maintainability for the problem at hand.
