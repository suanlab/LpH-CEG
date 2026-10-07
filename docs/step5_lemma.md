# The Step-5 proof obligation

**Status (2026-10-05): the §6 proof is now the paper's proof of Theorem 1
(`docs/paper/extended_version.tex`, Lemmas 1–3). It has been checked step by
step by two independent model reviews (one adversarial) and against the code,
and an adversarial search found no counterexample (§7) — but it has NOT been
refereed by a human theorist. That sign-off is still requested; if it finds a
gap, the paper must return to "verified, not proved".**
**Audience: the external theorist asked to sign off on Theorem 1.**

This note isolates the one unproven step in the Composition Inequality,
states it as a self-contained lemma, proves the easy case, explains why the
easy case is exactly the *uninteresting* one, reports what a machine
check has settled, and (§6–7) records the general proof and its review.

---

## 1. What Steps 1–4 give, and what is left

Fix a hyperpath $P = (J_0 = \emptyset, B_1, J_1, \dots, B_m, J_m = \mathcal{E})$.
Write

- $A_j = \mathrm{attr}(B_j)$ — all attributes of the $j$-th block,
- $V_{j} = \mathrm{attr}(J_{j})$ — attributes bound after $j$ steps,
- $I_j = A_j \cap V_{j-1}$ — the interface,
- $E_j = A_j \setminus I_j$ — the attributes *first bound* at step $j$.

Two structural facts are used throughout and are immediate from the
definition of a hyperpath:

> **(F1)** $E_j = V_j \setminus V_{j-1}$, hence $\{E_j\}_{j=1}^m$ **partitions** $V$.
> **(F2)** $I_j \subseteq V_{j-1}$.

Step 3 supplies, for every step $j$ and every $p \in \mathcal{P}$, the
$\ell_p$ degree-sequence inequality

$$\tfrac{1}{p} h(I_j) \;+\; h(E_j \mid I_j) \;\le\; \log_2 \mu_p(B_j \mid I_j). \tag{S3}$$

Step 4 multiplies (S3) by $w_{j,p} \ge 0$ and sums. So the bound
$\log_2 |Q(D)| \le \sum_{j,p} w_{j,p} \log_2 \mu_p(B_j \mid I_j)$ follows
**iff** we can close the gap with:

> **Obligation (O).** For every admissible $h$,
> $$\sum_{j,p} w_{j,p}\Bigl[\tfrac{1}{p} h(I_j) + h(E_j \mid I_j)\Bigr] \;\ge\; h(V).$$

`extended_proofs.tex` Step 5 asserts (O) "by subadditivity of entropy,
since every attribute's entropy is covered with total weight $\ge 1$".
**That sentence is not a proof.** Subadditivity gives $h(I_j) \le \sum_{a \in I_j} h(a)$,
which is an *upper* bound on a term appearing on the *lower*-bounded side —
the wrong direction. The real content is §3 below.

---

## 2. Lemma A — the per-step case (proved)

> **Lemma A.** If $\sum_{p} w_{j,p} \ge 1$ for **every step** $j$, then (O)
> holds for every monotone, submodular $h$ with $h(\emptyset)=0$ (in
> particular for every entropic $h$).

*Proof.* All terms below are non-negative because $h$ is monotone.

$$\sum_{j,p} w_{j,p}\Bigl[\tfrac{1}{p}h(I_j) + h(E_j\mid I_j)\Bigr]
\;\ge\; \sum_{j,p} w_{j,p}\, h(E_j \mid I_j)
\;=\; \sum_{j} \Bigl(\sum_p w_{j,p}\Bigr) h(E_j \mid I_j)$$

dropping the non-negative $\tfrac1p h(I_j)$ terms. By hypothesis
$\sum_p w_{j,p} \ge 1$ and $h(E_j\mid I_j) \ge 0$, so this is
$\ge \sum_j h(E_j \mid I_j)$. By (F2) $I_j \subseteq V_{j-1}$, and
conditioning reduces entropy, so $h(E_j \mid I_j) \ge h(E_j \mid V_{j-1})$.
Hence the sum is $\ge \sum_j h(E_j \mid V_{j-1})$, which by (F1) is exactly
the chain rule along the hyperpath:
$\sum_j h(V_j \setminus V_{j-1} \mid V_{j-1}) = h(V)$. $\square$

Lemma A is elementary and needs only monotonicity, submodularity and the
chain rule. **It is not the constraint the paper or the code uses.**

---

## 3. Why Lemma A is not enough: the cover constraint is strictly weaker

`src/estimators/lp_bound.py::_attribute_coverage` implements an
**attribute-wise** constraint: for each attribute $a \in V$,

$$\sum_{j,p} \mathrm{cov}(a,j,p)\, w_{j,p} \ge 1,
\qquad
\mathrm{cov}(a,j,p) = \begin{cases}
1 & a \in E_j \\[2pt]
1/p & a \in I_j \\[2pt]
0 & \text{otherwise.}
\end{cases} \tag{C}$$

By (F1) each $a$ lies in $E_{j(a)}$ for exactly one step $j(a)$, so (C) reads

$$\sum_p w_{j(a),p} \;+\; \sum_{j' \,:\, a \in I_{j'}} \sum_p \tfrac{1}{p} w_{j',p} \;\ge\; 1 .$$

The second sum is the **redundant coverage** an attribute receives by
reappearing in later interfaces. It lets $\sum_p w_{j(a),p}$ fall *below* 1,
so **(C) does not imply the hypothesis of Lemma A.** This is not a
technicality — it is where the method's improvement comes from. Measured
LP optima (`shannon_implication_check.py`, 60 vertices / 240 edges):

| shape | per-step $\sum_p w_{j,p}$ | min |
|---|---|---|
| cycle-4 | 0.667, 0.667, 0.667, 0.333 | **0.333** |
| cycle-6 | 0.667 ×5, 0.333 | **0.333** |
| $K_4$ | 0.5, 1.0, 0.0, 1.0, 0.0, 0.0 | **0.0** |
| star-3 (acyclic) | 1.0, 1.0, 1.0 | 1.0 |

> **The gap is exactly the contribution.** On acyclic queries the LP is
> forced back to per-step weight 1 — Lemma A applies, and the paper
> correspondingly reports *no improvement*. On cyclic queries the weights
> drop to 1/3 (or 0), Lemma A does **not** apply, and that is precisely
> where the reported tightening occurs. Any proof of (O) under (C) must
> therefore use the redundant-coverage structure; no weakening of the
> constraint can rescue it without destroying the result.

---

## 4. What has been mechanically settled

For fixed weights, (O) is *linear in $h$*. So over the polymatroid
(Shannon) cone it is decidable by linear programming:

$$\min_h \;\; \sum_{j,p} w_{j,p}\bigl[h(A_j) - (1-\tfrac1p)h(I_j)\bigr] - h(V)
\quad \text{s.t. } h \text{ polymatroid},\; h(\emptyset)=0,\; h(V)=1 .$$

A minimum $\ge 0$ certifies (O) for **all** databases at those weights, via
Shannon inequalities alone. A negative minimum exhibits a polymatroid
witness and proves no Shannon-type derivation exists.

Moreover the check can be made **complete in the weights**. (O) is affine
in $w$; the feasible set $W = \{w \ge 0 : \text{(C)}\}$ has recession cone
$\mathbb{R}^N_{\ge 0}$ (as $\mathrm{cov} \ge 0$ entrywise), whose extreme rays
are unit vectors, each of which satisfies (O)'s left side $\ge 0$ termwise.
Since the set of certified $w$ is convex and closed under non-negative
combination, checking **every vertex of $W$** plus those rays decides (O)
for all of $W$.

`scripts/shannon_implication_check.py` does this. Result
(`results/validity/shannon_implication.csv`, 72 checks):

| shape | $|V|$ | vertices of $W$ | min slack | verdict |
|---|---:|---:|---:|---|
| triangle | 3 | 31 | 0.000000 | holds ∀ feasible $w$ |
| cycle-4 | 4 | 52 | 0.000000 | holds ∀ feasible $w$ |
| cycle-5 | 5 | 233 | 0.000000 | holds ∀ feasible $w$ |
| cycle-6 | 6 | 683 | 0.000000 | holds ∀ feasible $w$ |
| $K_4$ | 4 | 148 | 0.000000 | holds ∀ feasible $w$ |
| diamond | 4 | 110 | 0.000000 | holds ∀ feasible $w$ |
| path-4 | 4 | 24 | 0.000000 | holds ∀ feasible $w$ |
| star-3 | 4 | 27 | 0.000000 | holds ∀ feasible $w$ |
| chordal-5 | 5 | 409 | 0.000000 | holds ∀ feasible $w$ |
| $K_5$ | 5 | 1{,}995 | 0.000000 | holds ∀ feasible $w$ |
| chordal-6 | 6 | 883 | 0.000000 | holds ∀ feasible $w$ |
| cycle-7 | 7 | 2{,}284 | 0.000000 | holds ∀ feasible $w$ |

Vertex enumeration terminated below its cap in every case, so these rows
are exhaustive, not sampled.

**The minimum is exactly $0$, never positive.** The obligation is tight:
some polymatroid attains equality. This independently reproduces the
empirical finding $\min(\text{estimate}/\text{true}) = 1.0000$ over 207,000
databases — two different methods agreeing that the bound is tight but
never violated.

### Exactly what this does and does not establish

- **Does:** for these twelve (shape, hyperpath) pairs, (O) holds for *every*
  database and *every* cover-feasible weight vector, and is derivable from
  Shannon inequalities. For these shapes the defect in `extended_proofs.tex`
  is a **writing** defect, not a false claim.
- **Does not:** say anything about shapes, hyperpaths, or block sizes not
  in the table. It is a finite verification, not a general theorem. It also
  says nothing about $|V| \gtrsim 8$, where $2^{|V|}$ makes the LP costly.
- **Scope precision (important).** `shannon_implication_check.py` calls
  the estimator with `max_block_size=1` and takes the single returned
  hyperpath, so on its own it certifies (12 shapes) × (one hyperpath each)
  × (bs = 1). The widened check below removes the "one hyperpath" and
  "bs = 1" qualifiers for $|V| \le 6$.

### Widened certificate: every hyperpath, bs ∈ {1, 2, 3}

`scripts/shannon_check_wide.py` (output `results/validity/shannon_wide.csv`)
uses the fact that (O) depends on a hyperpath only through its step
structure — the sequence of (block attributes $A_j$, interface $I_j$). It
enumerates every ordered partition of the relations into connected blocks
of size $\le$ bs, de-duplicates by structure, and runs the complete
vertex-plus-rays check on each. Enumeration is capped at 400 structures
and 30 min per (shape, bs); the CSV records whether each pair was
finished (`complete`).

Result (2026-09-03): **6,197 structures, 1,733,414 vertices, worst slack
$0$, no violation** on all 33 (shape, bs) pairs for the eleven shapes with
$|V| \le 6$. Nineteen pairs are *complete* — every hyperpath the shape
admits at that block size was checked: triangle, 4-cycle, 5-cycle, path-4,
star-3 (bs = 1, 2, 3 each); diamond (bs = 1, 2); 6-cycle and chordal-5
(bs = 1). The other fourteen ($K_4$, $K_5$, chordal-6, and the larger
block sizes of diamond/6-cycle/chordal-5) hit the structure cap or the
time budget; they are a prefix of the enumeration, not a sample, and they
too show no violation. So the certified region is now

  (every hyperpath) × (bs ≤ 3) on 19 (shape, bs) pairs, plus 3,000+
  further structures on the remaining 14 pairs — all at $|V| \le 6$ —
  plus the original one-hyperpath check of the 7-cycle.

---

## 5. What is asked of the theorist

1. **Is (O) true in general under (C)?** A proof must exploit redundant
   coverage (§3), since Lemma A provably cannot reach the cyclic case.
   A natural route: show (C) implies the existence of a *fractional chain
   decomposition* assigning each $a$'s chain-rule term
   $h(a \mid \text{preds})$ to covering steps with total weight $\ge 1$,
   where an interface occurrence of $a$ at step $j'$ can absorb at most a
   $1/p$ share of it. The $\mu_p$ inequality (S3) is what should license
   that $1/p$ share; making this precise is the crux.
2. **If it is false**, the machine check says any counterexample needs
   $|V| \ge 7$, a shape outside the table, or (for $K_4$, $K_5$,
   chordal-6 and the larger block sizes of diamond / 6-cycle / chordal-5)
   a hyperpath beyond the enumerated prefix — a useful search restriction.
   The counterexample search should then be re-aimed there.
3. **If it is true but needs an extra hypothesis**, please state it so it
   can be added to the LP as a constraint and re-measured; §3's table shows
   what such a hypothesis would cost (any constraint forcing per-step
   weight to 1 removes the entire cyclic improvement).

Until (1) is settled, the paper must keep describing Theorem 1's validity as
supported by exhaustive finite verification rather than by a general proof.

---

## 6. The general proof — model-reviewed, awaiting human sign-off

An adversarial review of this note produced the following argument, which
would make (O) a theorem for **arbitrary $|V|$, arbitrary hyperpaths,
arbitrary block sizes and arbitrary $p \ge 1$**, retiring the finite
verification as the load-bearing evidence. It has **not** been checked by
a human theorist, so the paper does not yet claim it. **This section is
the specific thing we are asking you to referee.**

### Reduction

Both (O) and (C) depend on $w$ only through
$$w_j := \sum_p w_{j,p}, \qquad s_j := \sum_p \tfrac{1}{p} w_{j,p},
\qquad 0 \le s_j \le w_j \ \ (\text{since } p \ge 1).$$
(O) becomes $\sum_j \bigl[s_j\, h(I_j) + w_j\, h(E_j \mid I_j)\bigr] \ge h(V)$,
and (C) becomes, for each attribute $a$,
$w_{j(a)} + \sum_{j' : a \in I_{j'}} s_{j'} \ge 1$.
Put $\delta_j := \max(0,\, 1 - w_j)$ and $V_j := E_1 \cup \dots \cup E_j$.

### Three steps

1. **(Interface terms.)** For any set $I$, the chain rule along the ordered
   partition $\{E_j\}$ gives $h(I) = \sum_j h(I \cap E_j \mid I \cap V_{j-1})
   \ge \sum_j h(I \cap E_j \mid V_{j-1})$, using $I \cap V_{j-1} \subseteq V_{j-1}$
   and submodularity. Apply with $I = I_{j'}$.
2. **(Fractional subadditivity.)** Fix $j$ with $\delta_j > 0$. By (C), every
   $a \in E_j$ satisfies $\sum_{j' : a \in I_{j'}} s_{j'} \ge 1 - w_j = \delta_j$,
   so $\{I_{j'} \cap E_j\}$ with weights $s_{j'}/\delta_j$ is a **fractional
   cover** of $E_j$. The set function $f(S) = h(S \mid V_{j-1})$ is monotone
   submodular with $f(\emptyset)=0$, hence fractionally subadditive, giving
   $\sum_{j'} s_{j'}\, h(I_{j'} \cap E_j \mid V_{j-1}) \ge \delta_j\, h(E_j \mid V_{j-1})$.
3. **(Extension terms.)** $I_j \subseteq V_{j-1}$ by (F2) and
   $w_j \ge 1 - \delta_j$, so $w_j\, h(E_j \mid I_j) \ge (1-\delta_j)\, h(E_j \mid V_{j-1})$.

Summing (1)+(2) over $j', j$ and adding (3) gives
$\sum_j [\delta_j + (1-\delta_j)]\, h(E_j \mid V_{j-1}) = \sum_j h(E_j \mid V_{j-1}) = h(V)$
by the chain rule. $\square$

### Why this is the right shape of argument

§3 argued that any proof must exploit redundant coverage. Step 2 is exactly
that: redundant coverage is what makes $\{I_{j'} \cap E_j\}$ a fractional
cover of $E_j$ with total weight $\ge \delta_j$. Fractional subadditivity of
$S \mapsto h(S \mid V_{j-1})$ is what licenses the $1/p$ interface share that
§1 could not account for. It also explains the exact-$0$ slack of §4:
step (3) is tight when $I_j = V_{j-1}$, and step (2) is tight for modular $h$.

### Evidence, and what we still do not have

- Only monotonicity and submodularity are used, so the argument, if correct,
  holds over the whole polymatroid cone (a fortiori for entropic $h$).
- An independent search over **198 abstract hyperpath schemes** — random
  ordered partitions $\{E_j\}$ with arbitrary $I_j \subseteq V_{j-1}$ plus
  random closing steps, $|V| \in \{3,4\}$, enumerating all vertices of
  $\{(w,s) : 0 \le s \le w,\ \text{(C)}\}$ — found **no violation**, worst
  slack $0.0$. Nothing in that search is specific to our 12 families.
- **What we do not have:** a human theorist's confirmation. The one
  non-elementary ingredient is that monotone submodular functions are
  fractionally subadditive; if that is applied correctly here, the argument
  appears complete, and if it is not, step 2 is where it fails.

---

## 7. Review record

- **2026-10-05, adversarial model review (verdict: CORRECT).** Each step
  checked for every polymatroid: Step 1 (chain rule + conditioning, using
  $E_j \cap V_{j-1} = \emptyset$), Step 2 (fractional subadditivity of
  $S \mapsto h(S \mid V_{j-1})$, with a self-contained proof now in the
  extended version as Lemma 2), Step 3, and the final sum including
  $\delta_j = 0$. `lp_bound.py::_attribute_coverage` implements (C)
  exactly (1 on $E_j$, $1/p$ on $I_j$, 0 at $p = \infty$; $E_j \cap I_j =
  \emptyset$ by construction), and interfaces are $\mathrm{attr}(J_{j-1})
  \cap A_j$ as the proof assumes. A random search over 900 abstract
  hyperpath schemes ($|V| \in \{3,4,5\}$, arbitrary interfaces, extra
  closing steps, $s_j > w_j$ allowed) found worst slack exactly 0.
- **Conditions now stated in the theorem.** Exact statistics of the database
  being bounded (or deterministic upper bounds on them); the sketched
  common-neighbour $\ell_2$ is outside the guarantee. (C) must hold exactly
  (LP solvers can miss it by ~1e-7; rescale). An empty block join gives 0.
- **Side finding while integrating.** The paper's Cyclic Improvement theorem
  claimed a per-edge optimal exponent ($w_j = p_j/(p_j+1)$ chosen per edge).
  That point is infeasible in general — the cover constraint between edges
  $j, j+1$ needs $p_{j+1} \le p_j$ (e.g. $p_j = 1, p_{j+1} = \infty$ covers
  the shared attribute only to 1/2). The theorem now states the bound for a
  common $p$ (each such point is tight), and the extended version's old
  acyclic step ("paths force unit weights"), which contradicted the paper's
  own 4-path example, was replaced.
- **Still requested from a human theorist:** read Lemmas 1–3 and the
  Composition theorem in `docs/paper/extended_version.pdf` (one page).
