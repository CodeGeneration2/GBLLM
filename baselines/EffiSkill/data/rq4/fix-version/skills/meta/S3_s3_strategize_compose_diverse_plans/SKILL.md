---
skill_id: S3
type: meta
language: python
name: S3 – Strategize (Compose Diverse Plans)
description: Design 2–4 meaningfully different solution plans by recombining known skills, deliberately varying the core levers
  (algorithm choice, math reformulation, precomputation, constant-factor tuning, etc.), and anticipating trade-offs.
---

## What S3 (Strategize) Does

S3 turns a bag of individual skills into **2–4 concrete, diverse plans** for solving a problem. Instead of jumping to the first idea, you:

1. Identify the **main optimization levers** available (math reformulation, algorithmic change, data structures, precomputation, constant-factor cuts, etc.).
2. Compose these levers into several **coherent strategies**, each with a distinct core idea.
3. Quickly assess **trade-offs** (complexity, implementation risk, corner cases).
4. Select 1–2 plans to attempt first, keeping the others as backups.

The examples in the weighted traces show many such levers:
- Switching from explicit combinatorics to a **parity or counting argument**.
- Replacing generic graph search with a **closed-form distance formula** for a structured graph.
- Trading heavy precomputation for **direct binomial** computations when the number of queries is small.
- Keeping big-O the same but achieving **large constant-factor reductions** by simplifying I/O or data structures.

S3 is the skill of **deliberately packaging these into alternative plans** rather than discovering them accidentally.

---

## Step 1 – Extract the Core Levers

Given a problem and your known tools, first list the “levers” you could pull. Use the traces as a menu:

1. **Asymptotic Algorithm Change**  
   - Examples from traces: BFS → O(1) distance formula over all pairs; factorial sums → parity argument.  
   - Questions:
     - Can I replace generic graph algorithms with problem-specific formulas?
     - Can I move from quadratic binomial summations to O(n) counting?

2. **Mathematical Reformulation**  
   - Examples: subset parity via presence_of_odd; moves on a grid via linear equations and one binomial; factorization-based counting via primes.  
   - Questions:
     - Does only the **parity**, divisibility, or some small feature of the numbers matter?
     - Can I express the problem as **paths**, **combinations**, or **prime factor** distributions?

3. **Precomputation vs. Direct Computation**  
   - Examples: removing factorial tables and using direct nCk; avoiding full 1..A_max sieves and doing input-driven factorization.  
   - Questions:
     - Will I do many similar operations so precomputation pays off?
     - Or are operations few/small so a direct O(k) approach is better than an O(N) table?

4. **Constant-Factor Reductions (Same Big-O)**  
   - Examples: simplifying I/O helpers; replacing Counter with dict; eliminating a suffix array in favor of a running sum.  
   - Questions:
     - Is the algorithm already optimal asymptotically, but bloated?
     - Can I remove layers (extra functions, data structures, conversions)?

5. **Early Termination / Focused Processing**  
   - Examples: only iterating over actual divisors instead of all up to sqrt(N); input-driven prime marking with early conflict detection.  
   - Questions:
     - Can I iterate only over **candidates that can actually work** (divisors, factors, edges)?
     - Can I **short-circuit** as soon as a condition is violated or satisfied?

Use this list as a small checklist. For any problem, mark which levers might be available.

---

## Step 2 – Define 2–4 Distinct Plan Archetypes

Now map those levers into **plan archetypes** you can reuse.

### Archetype A – Direct Algorithmic Optimization

**Core idea:** Replace generic algorithms with specialized ones or with a simpler iteration pattern.

- Typical moves:
  - Graph: BFS-from-every-node → direct distance formula + pair iteration.  
    (Seen in the traces where a path+one-edge graph is handled in O(N²) with pure arithmetic.)
  - Summations: double loops → prefix sums or hashmap-based counting.  
    (E.g., zero-sum subarrays via prefix sums and a frequency map.)
- When to consider: data has special structure (line graph, tree, simple grid).

**Plan template:**
1. Identify structure (e.g., graph shape, array pattern).
2. Derive O(1) or O(log N) computation for the basic unit (distance, segment, combination).
3. Iterate over elements/pairs with that formula.
4. Keep counters/arrays for aggregated results.

---

### Archetype B – Mathematical Reframing

**Core idea:** Replace brute-force combinatorics or counting with a closed-form or much simpler reasoning.

- Typical moves:
  - Subset-sum parity: instead of summing C(n,k), notice **only parity** matters; half the subsets are even if there is any odd element.
  - Move-count problems: solve a **linear system** for move counts and use exactly one binomial coefficient.
  - Factorization problems: express counts via **prime exponents** and products of binomials.
- When to consider: problem mentions combinations, parity, divisibility, or constrained moves.

**Plan template:**
1. Abstract the problem: what is really being chosen? subsets, paths, multisets of primes?
2. Express constraints algebraically (equalities, parity, exponents).
3. Derive a formula or small set of formulas (e.g., nCk, 2^(n-1)).
4. Implement with attention to numeric issues (modulo, overflow, integer vs float).

---

### Archetype C – Precompute vs Direct Trade-off

**Core idea:** Decide whether to precompute tables or compute each query directly.

- Typical moves:
  - Remove factorial / inverse factorial arrays in favor of direct nCk when only a few small-k queries exist.
  - Avoid full sieves over [1..A_max] when factoring only the inputs is enough.
- When to consider: N is large but **number of queries or parameters (like exponent e)** is small.

**Plan template:**
1. Estimate: how many times will I call this operation, and how big are its parameters?
2. If many calls and parameters are large → precompute.  
   If few calls and parameters are small → direct computation.
3. Implement the chosen style:
   - Precompute: build arrays once, ensure O(1) queries.
   - Direct: implement a simple O(k) method using integer arithmetic.

---

### Archetype D – Constant-Factor Streamlining

**Core idea:** Keep the asymptotic algorithm but make it lean.

- Typical moves:
  - Remove unused helpers and extra wrapper functions.
  - Use a single pass instead of two when possible (prefix vs suffix sums).
  - Replace heavy data structures (e.g., Counter) with lighter ones (dict) in tight loops.
- When to consider: algorithm is already asymptotically optimal but too slow in practice, or the problem is small but performance is still critical.

**Plan template:**
1. Identify the main hot loop.
2. Remove layers in that loop (function calls, conversions, complex objects).
3. Re-express logic in a **single linear scan** if possible.
4. Re-check correctness: no off-by-one, no lost edge cases.

---

## Step 3 – Compose 2–4 Concrete Plans for a New Problem

Given a new problem, apply S3 as follows.

### 1. Quick Lever Scan

In 1–2 minutes, answer:
- Is there recognizable **structure** (line graph, tree, grid, subset, multiset of primes)?
- Are there **combinatorial or parity** hints?
- Are **constraints** big enough that O(n²) or heavy precomputation is risky?

Check against the lever list from Step 1.

### 2. Draft Diverse Plans (Example Pattern)

For illustration, suppose you face a “count paths / ways” problem with moderate constraints.

You might produce:

- **Plan 1 – Direct DP/Graph Approach (Archetype A + D)**
  - Model states explicitly; use DP or BFS/DFS.
  - Optimize with arrays and simple loops, avoid heavy containers.
  - Pros: straightforward, easy to reason about.
  - Cons: may be O(N²) or heavier.

- **Plan 2 – Combinatorial / Math Formula (Archetype B)**
  - Recognize a path-count equals a binomial coefficient or similar.
  - Use direct nCk (or precomputed factorials, depending on constraints).
  - Pros: much faster, often O(1) or O(n).
  - Cons: requires correct derivation; more risk of off-by-one or feasibility mistakes.

- **Plan 3 – Hybrid with Limited Precomputation (Archetype B + C)**
  - Use mathematical simplification to reduce to a small number of nCk or factor operations.
  - Choose between direct or table-based binomials based on n and number of queries.
  - Pros: balances speed with implementation effort.
  - Cons: requires accurate constraint reading and a small design decision.

You now have three **meaningfully different** routes, not just small tweaks.

### 3. Evaluate and Pick an Order

For each candidate plan, quickly note:
- **Time/space complexity** and constant factors.
- **Implementation complexity**: how error-prone is it?
- **Edge cases**: are there numerically delicate parts (mod, divisibility checks)?

Typical choice patterns:
- If a math formulation is clear → try the **math plan first**, keep DP as backup.
- If constraints are borderline → avoid obviously heavy (O(N² log N) with big constants) plans.
- If time is low or you are unsure about the math → implement the **simple DP/graph plan**, then refine.

---

## Step 4 – Sanity Check the Chosen Plan Using the Others

Even after choosing one plan, you can use the *ideas* from the others to validate and refine it.

- Use a **simpler plan** (e.g., brute-force or DP) on small random tests to verify the fast formula-based plan.
- Borrow constant-factor improvements from Plan D even if you use Plan B or C.
- Use Plan B’s math reasoning to simplify or bound parts of Plan A (e.g., knowing that distances can only be within 1..N-1, or that half the subsets should match a parity constraint).

This reflex of cross-checking plans is part of S3.

---

## Quick S3 Checklist

When facing a new problem, do this:

1. **List levers**: structure? math? precomputation? constant-factor? early exit?
2. **Sketch 2–4 plans**, each dominated by a *different* lever or archetype.
3. **Estimate** costs and risks for each.
4. **Pick an execution order** (fastest/clearest first), keep backups in mind.
5. **Cross-check** the chosen plan’s outputs against a simpler plan on small tests.

Practicing this loop will make your solution attempts more robust and your optimizations more systematic, as illustrated by the varied transformations in the weighted traces.
