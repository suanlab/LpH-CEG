# Datasets

Real-data experiments read from this `data/` directory. Each subdirectory
is currently a **symlink** to a sibling repository on the original authors'
machine; for an independent reproduction, replace each symlink with the
actual dataset (or re-create the symlink) at the path shown below.

Scripts resolve this directory **repo-relative** (`<repo>/data/...`) and
honor the `LPH_DATA_DIR` environment variable to point elsewhere:

```bash
LPH_DATA_DIR=/path/to/datasets PYTHONPATH=. python scripts/job_experiment.py
```

Synthetic power-law experiments (cycle-k scalability, skew sensitivity)
generate their data in-process with fixed seeds and need **no download**.

## Expected layout

| Subdir         | Used by                                   | Source |
|----------------|-------------------------------------------|--------|
| `hetionet/`    | every Hetionet experiment and suite | Hetionet v1.0 (Himmelstein et al., *eLife* 2017; **CC0**, https://github.com/hetio/hetionet). The 13 per-metaedge edge lists used here are **shipped** in `data/hetionet_cc0/` with `SHA256SUMS`; use them as `data/hetionet/` (`ln -s hetionet_cc0 data/hetionet`). Row order matters: suites cap relations at the first N rows in file order. |
| `epinions/`    | real-data cycle/triangle                  | Epinions social/trust network — SNAP `soc-Epinions1`. https://snap.stanford.edu/data/soc-Epinions1.html |
| `imdb/`        | co-actor cycle queries                    | IMDb non-commercial datasets. https://datasets.imdbws.com/ |
| `job_medium/`  | `job_experiment.py`, JOB tables           | Join Order Benchmark (IMDb subset) — Leis et al. https://github.com/gregrahn/join-order-benchmark |
| `gcare/`      | G-CARE cyclic suite (`gcare_suite.py`) | G-CARE (Park et al., SIGMOD 2020), https://github.com/yspark-dblab/gcare: download `dataset.tar.gz`, `queryset.tar.gz` and `true_cardinality.tar.gz` (Google Drive links in its README) into `data/gcare/`; unpack the datasets in place, the queries into `queries/` and the true cardinalities into `truth/` (they share file names). `scripts/gcare_prepare.py` loads AIDS and YAGO into `gcare.duckdb`. |
| `imdb_cycles/` | IMDb cycle suite (`evaluate_imdb_suite.py`) | **Derived, not downloaded**: `scripts/prepare_imdb_cycles.py` projects JOB's full IMDb snapshot (`cast_info` by role, `movie_companies` by company type, `movie_keyword`) onto the 200 most-voted feature films (an earlier 2005 slice was too sparse; see the script docstring). Put the JOB CSVs in `data/job_full/` or point `LPH_JOB_DIR` at them; `MANIFEST.json` pins each output file's sha256, and evaluation refuses to run on a mismatch. |

## File format

Each relation is a CSV/TSV the loader reads as `(schema, rows)`; see the
`load_*` helpers in the consuming script for the exact column mapping
(e.g. `job_experiment.py:load_csv_relation`). Edge relations are expected
as two-column `(src, dst)` integer/string vertex identifiers.

> Note: these datasets are external and not redistributed here. The
> reproducibility manifest is `requirements.txt` / `pyproject.toml`;
> one-click reproduction is `scripts/reproduce_all.sh`.
