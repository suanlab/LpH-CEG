# LpH-CEG — block ℓp-norm statistics for cyclic join bounds

LpH-CEG computes **guaranteed upper bounds** on the cardinality of cyclic
joins from ℓp-norms of degree sequences, and asks where the remaining
headroom lies: in the *program* that combines the statistics, or in the
*statistics* themselves. On single-relation statistics LpBound's
polynomial flow LP already reaches the polymatroid optimum, and our
fractional block-cover LP is dominated. Block statistics (degree sequences
of two-relation joins, conditioned on their interfaces) tighten LpBound by
a median 46–920× on registered Hetionet cycles, much less on IMDb unless the
polymatroid program uses them. The optional ℓp flow relaxation is a
candidate, checked against that program rather than generally proved valid.
On G-CARE's AIDS the program gap dominates. Paper: `docs/paper/main.tex`.

**Status: research artifact under submission (ICDE 2027); public copy at
<https://github.com/suanlab/LpH-CEG>.** The validity
theorem (the lp Composition Inequality) is proved in
[`docs/paper/extended_version.pdf`](docs/paper/extended_version.pdf); the
proof was checked by independent model reviews and by an exhaustive LP
certificate, but has **not yet been refereed by a human theorist** — see
[`docs/step5_lemma.md`](docs/step5_lemma.md).

## Install

```bash
pip install -r requirements.lock     # exact pinned versions (Python 3.10)
```

`requirements.txt` is a looser convenience list and omits `duckdb`/`kuzu`,
which the end-to-end engine experiments need. Prefer the lockfile. A
`Dockerfile` is provided; note it does not install LaTeX, so use a target
that excludes the paper build inside the container.

## The three commands

```bash
PYTHONPATH=. pytest tests/ -q            # data-dependent tests report missing input explicitly
bash scripts/reproduce_all.sh smoke      # tests + lint + figures + paper (~1 min)
bash scripts/reproduce_all.sh all        # every experiment (~hours; needs data/)
```

Individual targets: `tests`, `lint`, `figures`, `paper`, `experiments`.

## What reproduces without any data download

The **synthetic half runs standalone** — fixed seeds, no external input, and
fixed-seed result values can be checked; floating-point results and timings may vary across environments. This covers the validity search, the Shannon-cone
certification, the ablations, the gap characterization, and the LP-cost
measurements.

The **real-data half**:

- **Hetionet** edge lists are shipped (the upstream project dedicates its original content to CC0; underlying source terms must also be checked): the 13 per-metaedge edge lists the suites
  use are in `data/hetionet_cc0/` with `SHA256SUMS`. Link them as
  `data/hetionet` (`ln -s hetionet_cc0 data/hetionet`). Row order matters —
  relations are capped at the first N rows in file order.
- **IMDb** (Join Order Benchmark snapshot) is not redistributed. Download the
  JOB CSVs (https://github.com/gregrahn/join-order-benchmark) into
  `data/job_full/` (or set `LPH_JOB_DIR`), then
  `python scripts/prepare_imdb_cycles.py` derives the suite's relations;
  `evaluate_imdb_suite.py` refuses to run unless their sha256s match the
  registry.
- **Epinions** (SNAP `soc-Epinions1`) is a public download; see
  `data/README.md`.

## How claims stay tied to evidence

| Gate | What it enforces |
|---|---|
| `scripts/claim_audit.py` | every registered numerical claim (the count is printed by the script) is checked against CSV-derived values; token occurrence is checked in both TeX files, not claim-context identity or completeness |
| `tests/unit/test_query_suite.py` | the 377-query suite must match its registered hash |
| `tests/unit/test_long_suite.py` | the 120-query long-cycle suite (|V| = 6, 8, 10) must match its hash and re-derive from its rule |
| `tests/unit/test_agm_pairjoin.py` | the AGM-over-pair-join-sizes baseline (`scripts/agm_pairjoin_baseline.py`): join sizes and cover LP sanity |
| `tests/unit/test_stats_catalogue.py` | the amortised statistics catalogue (`scripts/stats_catalogue.py`): relabel-invariant keys, orientation, catalogue hits reproduce the uncached bound |
| `tests/unit/test_shannon_wide.py` | hyperpath-structure enumeration behind the every-hyperpath Step-5 certificate (`scripts/shannon_check_wide.py`) |
| `tests/unit/test_validity.py` | the bound must never fall below a true count |
| `tests/unit/test_pushdown_stats.py` | join-free block statistics (`src/statistics/pushdown_stats.py`) equal the materialized ones; the codegree triangle bound is an upper bound; the l2 codegree alone recovers the bs=2 bound on the test instance (three norms, flow LP; at eleven norms the one-side interfaces matter too, `scripts/ablation_loo.py`) |
| `tests/unit/test_edge_matrix.py` | bag inputs give the set bounds; parallel atoms stay valid; `guaranteed_block_bound` falls back to the cover LP on size and on solver failure; eleven norms on degree-400 hubs stay finite and valid |
| `tests/unit/test_separation.py` | the extended version's separation example: every listed pairwise statistic and every tree-homomorphism count into one relation agree, the common-neighbour l2 (32 vs 24) does not |
| `tests/unit/test_flow_bound.py` | the flow LP (`src/estimators/flow_bound.py`): analytic path bound, ℓp bundle = polymatroid, bs=1 == entropic, every order valid and the best cyclic order tight |
| build gate in `reproduce_all.sh paper` | fails on any LaTeX error or unbalanced theorem env |
| `MANIFEST.sha256` | source-file integrity (`scripts/gen_manifest.sh` regenerates) |

Every suite is admitted by a fixed, data-blind rule (SHA-256 order of the
canonical relation sequence) and hash-pinned. The v1 suite was committed
before it was measured; the long and IMDb suites were committed together
with their first results, so for them the rule, not the commit order, is
the guarantee. Post-registration choices (the 109-query short subset, row
caps, G-CARE size caps) are disclosed in the paper.

## Reproduction boundaries

`estimate_lph_ceg` is the cover estimator; `guaranteed_block_bound` consumes
precomputed constraints and a valid cover fallback, returning the polymatroid
value up to 12 variables or the cover fallback on size/solver failure.
Marginal selectivity products are not guaranteed bounds: use filtered tuples.
The legacy shared H-CEG `estimate_pessimistic` selectivity option is outside
this guarantee; paper baseline runs use unit selectivities.

The full workflow also needs G-CARE and the public CorrBound artifact (see
`data/README.md`). CorrBound's LP rerun needs `ortools`, `sqlparse` and
`joblib`; set `LPH_CORRBOUND_PYTHON` to an existing environment containing
them and `LPH_CORRBOUND_DIR` to the artifact checkout. The `all` target is
many hours and is not validated by the short tests or the CSV audit. It first
checks for inputs this repository does not build and fails if any is missing:
the legacy bs=1 scripts (JOB-medium CSVs, 10k IMDb edge lists; they back only the
extended version's bs=1 JOB-medium paragraph) and the CorrBound environment. `LPH_SKIP_LEGACY=1` skips the
former and `LPH_REUSE_STORED_CORRBOUND=1` keeps the stored CorrBound LP results;
both are reported as skipped or reused in the log.

## Layout

- `src/estimators/` — `lph_ceg.py` (entry point), `lp_bound.py` (the cover LP),
  `lpbound_entropic.py` (the 2^|V| Shannon-cone comparator), `flow_bound.py`
  (polynomial flow LP over ℓp block constraints, after Im et al. / LpBound's
  LPflow; the order relaxation for |I|=2 blocks), `agm_bound.py`
- `src/statistics/lp_stats.py` (lp block statistics from the materialized
  block join), `pushdown_stats.py` (the seven join-free interfaces of a
  binary block from single-relation passes; the codegree l2 triangle bound
  and Hutchinson sketch)
- `src/core/`, `src/utils/`, `src/statistics/block_stats.py` — shared H-CEG
  machinery; edit these in the sibling `H-CEG` project, not here
- `scripts/` — one script per experiment; each writes a CSV under `results/`
- `scripts/gcare_semantics/` — one-off diagnostics of the G-CARE YAGO count gap (alternative matching rules, edge reversals, label substitutions; outputs in `results/gcare/semantics/`), kept as run; not part of the paper pipeline
- `benchmarks/query_suite_v1.json` — the pre-registered evaluation suite (|V| ≤ 5)
- `benchmarks/query_suite_v2_long.json` — the pre-registered long-cycle suite (|V| = 6, 8, 10; 40 per length by hash order)
- `docs/paper/` — the manuscript (`main.tex`); `docs/step5_lemma.md` — the historical proof-obligation note (the current proof is in the extended version)

## License

MIT (see [`LICENSE`](LICENSE)), matching LpBound's. The license covers this
repository, including the copies of the shared H-CEG core under `src/core/`,
`src/utils/`, `src/statistics/block_stats.py`, `src/estimators/pessimistic.py`
and `src/wcoj/{evaluation,residual_bound,variable_order}.py`. Hetionet dedicates its original content to CC0 but asks users to consider each integrated source's terms. The stripped edge lists here do not retain those per-edge license attributes, so this audit has not established blanket redistribution rights. Other datasets retain their source terms (see `data/README.md`).
