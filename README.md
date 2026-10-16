# LpH-CEG — block ℓp-norm statistics for cyclic join bounds

LpH-CEG computes **guaranteed upper bounds** on the cardinality of cyclic
joins from ℓp-norms of degree sequences, and asks where the remaining
headroom lies: in the *program* that combines the statistics, or in the
*statistics* themselves. On single-relation statistics LpBound's
polynomial flow LP already reaches the polymatroid optimum, and our
fractional block-cover LP is dominated. Block statistics (degree sequences
of two-relation joins, conditioned on their interfaces) tighten LpBound by
a median 46–920× on registered Hetionet cycles, much less on IMDb unless a
better program (our ℓp extension of the flow LP, certified per query by
the polymatroid bound) uses them, and on G-CARE's AIDS the program gap
dominates. Paper: `docs/paper/main.tex`.

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
PYTHONPATH=. pytest tests/ -q            # 127 tests (2 need the real data)
bash scripts/reproduce_all.sh smoke      # tests + lint + figures + paper (~1 min)
bash scripts/reproduce_all.sh all        # every experiment (~hours; needs data/)
```

Individual targets: `tests`, `lint`, `figures`, `paper`, `experiments`.

## What reproduces without any data download

The **synthetic half runs standalone** — fixed seeds, no external input, and
it reproduces bit-for-bit. This covers the validity search, the Shannon-cone
certification, the ablations, the gap characterization, and the LP-cost
measurements.

The **real-data half**:

- **Hetionet** (CC0) is shipped: the 13 per-metaedge edge lists the suites
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
| `scripts/claim_audit.py` | 436 headline numbers must match their source CSV cell (counts exactly), and reports pinned values missing from `main.tex` |
| `tests/unit/test_query_suite.py` | the 377-query suite must match its registered hash |
| `tests/unit/test_long_suite.py` | the 120-query long-cycle suite (|V| = 6, 8, 10) must match its hash and re-derive from its rule |
| `tests/unit/test_agm_pairjoin.py` | the AGM-over-pair-join-sizes baseline (`scripts/agm_pairjoin_baseline.py`): join sizes and cover LP sanity |
| `tests/unit/test_stats_catalogue.py` | the amortised statistics catalogue (`scripts/stats_catalogue.py`): relabel-invariant keys, orientation, catalogue hits reproduce the uncached bound |
| `tests/unit/test_shannon_wide.py` | hyperpath-structure enumeration behind the every-hyperpath Step-5 certificate (`scripts/shannon_check_wide.py`) |
| `tests/unit/test_validity.py` | the bound must never fall below a true count |
| `tests/unit/test_pushdown_stats.py` | join-free block statistics (`src/statistics/pushdown_stats.py`) equal the materialized ones; the codegree triangle bound is an upper bound; the l2 codegree alone recovers the bs=2 bound |
| `tests/unit/test_flow_bound.py` | the flow LP (`src/estimators/flow_bound.py`): analytic path bound, ℓp bundle = polymatroid, bs=1 == entropic, every order valid and the best cyclic order tight |
| build gate in `reproduce_all.sh paper` | fails on any LaTeX error or unbalanced theorem env |
| `MANIFEST.sha256` | source-file integrity (`scripts/gen_manifest.sh` regenerates) |

Each gate has been checked to actually *fail* when violated, not merely to
pass. The query suite is registered **before** measurement and hash-pinned,
so query selection cannot have followed the results.

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
- `benchmarks/query_suite_v1.json` — the pre-registered evaluation suite (|V| ≤ 5)
- `benchmarks/query_suite_v2_long.json` — the pre-registered long-cycle suite (|V| = 6, 8, 10; 40 per length by hash order)
- `docs/paper/` — the manuscript (`main.tex`); `docs/step5_lemma.md` — the open
  proof obligation

## License

MIT (see [`LICENSE`](LICENSE)), matching LpBound's. The license covers this
repository, including the copies of the shared H-CEG core under `src/core/`,
`src/utils/`, `src/statistics/block_stats.py`, `src/estimators/pessimistic.py`
and `src/wcoj/{evaluation,residual_bound,variable_order}.py`. The datasets are
not redistributed and keep their own terms (see `data/README.md`).
