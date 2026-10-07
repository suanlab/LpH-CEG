# Our entropic comparator vs. the official LpBound implementation

**Status: formulation verified against the published source. Timing not
verified — see "What this does not establish".**

Reviewers asked, reasonably, whether the entropic baseline we compare
against is a strawman: it is our own implementation, and the crossover
argument is the paper's, so an under-powered comparator would flatter us.
This note records a line-by-line correspondence against the official
artifact.

Source: `github.com/fdbresearch/LpBound` (MIT), the repository linked from
the paper's project page, commit `ee01ad5`. The relevant file is
`src/lpbound/cyclic/LPBuilder.py`.

## The correspondence

| Component | Official LpBound | Ours (`src/estimators/lpbound_entropic.py`) |
|---|---|---|
| Cone generators | `add_elemental_submodularities()` and `add_elemental_monotonicities()` (L211–215) | `elemental_inequalities()` — the same minimal generating set |
| Submodularity | `h(Z∪{x}) + h(Z∪{y}) − h(Z) − h(Z∪{x,y}) ≥ 0` (L559) | identical, over all `Z ⊆ V∖{x,y}` |
| ℓp statistic, finite p | `p·h_il − (p−1)·h_i ≤ p·log₂ μ_p` (L267) | `h(A) − (1−1/p)·h(I) ≤ log₂ μ_p` |
| ℓp statistic, p = ∞ | `h_il − h_i ≤ log₂ μ_∞` (L265) | same, with `1/p = 0` |

The two ℓp forms are the same inequality: dividing the official constraint
by `p` gives `h_il − (1−1/p)·h_i ≤ log₂ μ_p`. So both programs maximise
`h(V)` over the Shannon cone subject to the same ℓp degree-sequence
constraints, which is what the paper claims when it calls our comparator
"the genuine entropic LpBound".

## What this does not establish

- **We did not run their code.** Reproducing their pipeline needs the C++
  solver compiled and their datasets; the correspondence above is a
  reading of the source, not an execution.
- **Timing is therefore ours-vs-ours.** They solve with OR-Tools GLOP and
  ship a parallel C++ solver; we use scipy/HiGHS interior-point. Our
  reported entropic solve times (43 s at |V|=12) are what *our* faithful
  implementation costs, and a tuned C++ implementation would be faster.
  The paper should not, and does not, claim their system is slow — only
  that this formulation's cost grows as `2^|V|` while the cover LP's does
  not.
- **Their repository is the public subset.** The README states that full
  paper reproduction, including a commercial system, lives in a separate
  private repository.

## Incidental checks

- The bibliography's `note = {Best Paper Award}` on `zhang2025lpbound`,
  flagged as unverified during review, is correct: SIGMOD 2025 announced
  it among the Best Paper Awards.
- Their `competitors.md` sets up SafeBound among the baselines, which
  independently confirms it belongs in our related work — it was missing
  and has since been added.
