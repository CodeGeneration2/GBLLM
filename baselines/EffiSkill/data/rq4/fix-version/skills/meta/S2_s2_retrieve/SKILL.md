---
skill_id: S2
type: meta
language: python
name: S2 Retrieve
description: A meta-skill for selectively invoking optimization operators based on uncertainty signals and a dynamic skill-budget,
  using confidence-aware routing instead of blindly scanning the full catalog.
---

## Overview

S2-Retrieve is the meta-skill that decides **which concrete operators to call, and how many**, based on:

- **What looks wrong in the trace/code** (symptoms & triggers)
- **How confident we are** about the root cause
- **How much budget** we have for additional analysis or refactors

Instead of trying every operator, S2 builds a small, targeted shortlist and escalates only if needed.

---

## Core Loop

At a high level:

1. **Read the situation**
   - Extract key signals from the task/trace:
     - Performance symptom: TLE, high memory, or just "could be cleaner".
     - Structural hints: DP, graph, combinatorics, digit loops, etc.
     - Language/implementation hints: Python I/O, heavy math, large arrays, nested loops.

2. **Form a hypothesis cluster (not mentioned explicitly)**
   - Infer likely *categories* of problems:
     - "Graph BFS/DFS or shortest path heavy" → graph & constant-factor operators.
     - "DP table, many dimensions, or big arrays" → DP/state-compression operators.
     - "Factorials, binomials, primes" → combinatorics/number-theory operators.
     - "Big loops over integers, floor division, divisibility" → arithmetic/closed-form operators.
     - "Large fixed buffers or counting" → streaming/prefix/frequency operators.

3. **Assign per-operator confidence**
   - For each operator, compute a rough score from:
     - Trigger overlap with observed patterns.
     - Match between tags (e.g. `dynamic_programming`, `graphs`, `binomial_coefficient`) and the task.
     - Strength of language cues (Python-specific vs generic math).

4. **Apply the skill budget**
   - Define a dynamic budget `B` = number of operators we can afford to consider deeply.
   - Start small (e.g., `B = 2–3`) and increase only if:
     - No operator above a minimum-confidence threshold, or
     - Previous suggestions failed / are clearly mismatched.

5. **Route to the top operators**
   - Select the **top-k operators by confidence**, bounded by `B`.
   - For each selected operator, generate:
     - A quick *fit-check*: do triggers really match this instance?
     - A **minimal, targeted suggestion** (what to refactor, not a full rewrite).

6. **Observe feedback and adapt**
   - If later context shows our operator(s) missed the mark, or the user pushes back:
     - Decrease confidence in similar patterns.
     - Consider the next-highest operators that were just below the budget cutoff.

---

## Confidence-Aware Routing Heuristics

### 1. High-confidence, narrow symptom

When a pattern is **very clear**, spend almost all budget on the best-fitting operator and maybe one backup.

Examples:

- **Graph BFS/shortest path symptoms**
  - Many BFS/DFS runs, adjacency matrix, or repeated queue creation.
  - Route primarily to **O004** (graph structure & data-structure optimization).

- **Linear DP with tiny state**
  - 1D/2D DP, dependence only on last few steps.
  - Route primarily to **O029** (rolling state linear DP).

- **Large factorial tables or binomial loops**
  - Precomputed factorial arrays, big nCr, or heavy combinations.
  - Route primarily to one of:
    - **O014** (right-size/remove combinatorial precomputation), or
    - **O023** (modular binomial combinatorics), depending on whether a modulus is present.

In these cases:

- `B ≈ 2`
- Use 1 main operator + 1 nearby backup for a different angle on the same structure.

### 2. Medium-confidence, mixed symptom

When **multiple categories are plausible**:

- Spread the budget across a few families.
- Prefer operators that:
  - Are **language-specific** to the user’s environment (e.g., Python → **O001**, **O018**).
  - Attack the **dominant cost center** (e.g., DP dimension vs big-int factorials).

Examples of routing sets:

- DP with a suspicious extra dimension and nested scans:
  - **O010** (DP state reformulation & dimension reduction)
  - **O029** (rolling DP & recurrence compression)

- Numeric loops with divisibility, floor-division, and large ranges:
  - **O009** (arithmetic loop → closed form)
  - **O019** (divisor enumeration & GCD folding)

- Frequency/counting code with large fixed buffers:
  - **O006** (right-size & stream work)
  - **O021** (refactor redundant range loops and pairwise aggregations)

Here, a typical budget is `B ≈ 3–4` with confidence slightly spread.

### 3. Low-confidence, weak or noisy signals

When trace/code is **short, noisy, or high-level** and structure is unclear:

- Start with the most **generic, low-risk** operators:
  - **O001** (Python constant-factor optimization) for hot loops, heavy structures, or I/O.
  - **O018** (replace heavy math/library calls) for repeated `math.*`, `pow`, or tiny NumPy use.
- Do a **lightweight probe**: explain what to look for in the code to confirm applicability.
- Only after we see clarifying details do we escalate to more specialized operators.

Budget strategy:

- Initial `B ≈ 1–2` with broad operators.
- Increase `B` only after more code/trace is visible.

---

## Dynamic Skill Budgeting

The skill budget controls **how many operators we seriously engage** in one pass.

### Inputs to budget decision

- **Problem size & complexity**
  - Small, simple tasks → keep `B` small.
  - Large or multi-part tasks → allow a larger `B`, but still phased.

- **User intent & constraints**
  - "Just make it pass" → prioritize one high-impact operator.
  - "Teach me options / alternatives" → allow multiple operators with brief contrasts.

- **Response length constraints**
  - If the context or user asks for brevity, treat that as a **hard cap** on `B`.

### Budget adjustment rules

1. **Start minimal**
   - Default initial `B = 2–3` unless context demands more.
2. **Only expand when justified**
   - Expand `B` if:
     - The top operator has **moderate confidence**, not high.
     - We see **two distinct, plausible root causes**.
3. **Don’t pile on overlapping operators**
   - Avoid suggesting multiple operators that solve the same subproblem in nearly identical ways (e.g., multiple binomial-specific operators for a simple single nCr).

---

## Practical Routing Recipes

This section gives **compact routing patterns** you can apply quickly.

### A. Graph-like problems

Signals:
- BFS/DFS per node, adjacency matrix, repeated queue creation, or many shortest-path calls.

Route:
- Primary: **O004** (optimize graph algorithms and representation).
- If still slow but algorithm is already correct: add **O001** (constant-factor cleanup) to tune Python overhead.

### B. DP / state-based problems

Signals:
- n×m tables, knapsack-like recurrences, bitmask DP, extra dimensions.

Route:
- For **extra dimensions or scan-heavy transitions** → **O010**.
- For **linear-time DP in one dimension with small fixed state** → **O029**.
- If memory is the main issue → prioritize **O029**, then **O010** if structure still heavy.

### C. Factorials, binomials, and modular combinatorics

Signals:
- `factorial`, `nCr`, big integer growth, long loops over k in binomial helper.

Route:
- If many modular binomials or big n → **O023**.
- If factorial tables are precomputed to a fixed large MAX regardless of input → **O014**.
- If only a few small-k combinations, especially from prime exponents → **O003** or **O007** depending on context.

### D. Numeric loops & closed forms

Signals:
- Long loops over 1..N with simple arithmetic, especially involving `n // i`, sums over divisors, or linear constraints.

Route:
- Floor-division and divisor-like patterns → **O009**.
- Explicit loops over potential divisors or divisibility checks → **O019**.
- Triple loops with linear constraints on indices → **O016** or **O022**.

### E. Streaming and counting

Signals:
- Large fixed-size arrays, repeated passes, big zero-initialized buffers, scanning over whole ranges with sparse actual data.

Route:
- Frequency/count style, sparse keys → **O006**.
- Many nested scans, repeated global sums or pairwise stats → **O021**.
- Simple product with overflow/threshold checks → **O028**.

### F. Constant-factor / implementation cleanup

Signals:
- Python-specific overhead, heavy classes in hot loops, poor I/O patterns, repeated `math.*` or `pow` in large loops.

Route:
- Algorithm is asymptotically fine but slow in Python → **O001**.
- Many math/library calls in hot loops → **O018**.

---

## Confidence Refinement from Feedback

As the interaction progresses, S2 updates its routing:

1. **User rejects an operator as irrelevant**
   - Lower confidence for similar triggers in this context.
   - Prefer different families in the next round.

2. **User provides more code or error details**
   - Re-run the trigger matching with updated information.
   - Allow switching to a different primary operator even if it wasn’t in the first shortlist.

3. **Optimization works but leaves residual issues**
   - If performance is improved but not enough, escalate from structural to constant-factor operators or vice versa.

---

## How to Apply S2-Retrieve in Practice

When you need to choose operators:

1. **Summarize the problem** in a sentence (structure + symptom).
2. **Map to 1–2 likely families** (graph, DP, combinatorics, numeric loops, streaming, constant-factor).
3. **Pick the top operator** in that family whose triggers closely match.
4. **Optionally add 1–2 backups** if confidence is moderate.
5. **Stop there** unless later context proves you wrong.

This keeps routing precise, explainable, and responsive to both confidence and the available skill budget.
