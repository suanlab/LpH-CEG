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
| `hetionet/`    | `wcoj_e2e_experiment.py`, real-data cycle | Hetionet v1.0 biomedical knowledge graph — Himmelstein et al., *eLife* 2017. https://github.com/hetio/hetionet |
| `epinions/`    | real-data cycle/triangle                  | Epinions social/trust network — SNAP `soc-Epinions1`. https://snap.stanford.edu/data/soc-Epinions1.html |
| `imdb/`        | co-actor cycle queries                    | IMDb non-commercial datasets. https://datasets.imdbws.com/ |
| `job_medium/`  | `job_experiment.py`, JOB tables           | Join Order Benchmark (IMDb subset) — Leis et al. https://github.com/gregrahn/join-order-benchmark |
| `imdb_cycles/` | IMDb cycle suite (`evaluate_imdb_suite.py`) | **Derived, not downloaded**: `scripts/prepare_imdb_cycles.py` projects JOB's full IMDb snapshot (`cast_info` by role, `movie_companies` by company type, `movie_keyword`) onto the 200 most-voted feature films (an earlier 2005 slice was too sparse; see the script docstring). Put the JOB CSVs in `data/job_full/` or point `LPH_JOB_DIR` at them; `MANIFEST.json` pins each output file's sha256, and evaluation refuses to run on a mismatch. |

## File format

Each relation is a CSV/TSV the loader reads as `(schema, rows)`; see the
`load_*` helpers in the consuming script for the exact column mapping
(e.g. `job_experiment.py:load_csv_relation`). Edge relations are expected
as two-column `(src, dst)` integer/string vertex identifiers.

> Note: these datasets are external and not redistributed here. The
> reproducibility manifest is `requirements.txt` / `pyproject.toml`;
> one-click reproduction is `scripts/reproduce_all.sh`.
