# Datasets

Real-data experiments read from this `data/` directory. Directories may contain local files or symlinks. For an independent
reproduction, install each dataset (or link it) at the path shown below.

Most scripts resolve paths **repo-relative** (`<repo>/data/...`). Some,
including `job_experiment.py`, accept `LPH_DATA_DIR`; check the consuming
script before relying on this override. The IMDb preparation script uses
`LPH_JOB_DIR`, and the CorrBound scripts use `LPH_CORRBOUND_DIR`:

```bash
LPH_DATA_DIR=/path/to/datasets PYTHONPATH=. python scripts/job_experiment.py
```

Synthetic power-law experiments (cycle-k scalability, skew sensitivity)
generate their data in-process with fixed seeds and need **no download**.

## Expected layout

| Subdir         | Used by                                   | Source |
|----------------|-------------------------------------------|--------|
| `hetionet/`    | every Hetionet experiment and suite | Hetionet v1.0 (Himmelstein et al., *eLife* 2017; original repository content **CC0**, with underlying source terms, https://github.com/hetio/hetionet). The 13 per-metaedge edge lists used here are **shipped** in `data/hetionet_cc0/` with `SHA256SUMS`; use them as `data/hetionet/` (`ln -s hetionet_cc0 data/hetionet`). Row order matters: suites cap relations at the first N rows in file order. |
| `epinions/`    | real-data cycle/triangle                  | Epinions social/trust network — SNAP `soc-Epinions1`. https://snap.stanford.edu/data/soc-Epinions1.html |
| `imdb/`        | co-actor cycle queries                    | IMDb non-commercial datasets. https://datasets.imdbws.com/ |
| `job_medium/`  | `job_experiment.py`, JOB tables           | Join Order Benchmark (IMDb subset) — Leis et al. https://github.com/gregrahn/join-order-benchmark |
| `gcare/`      | G-CARE cyclic suite (`gcare_suite.py`) | G-CARE (Park et al., SIGMOD 2020), https://github.com/yspark-dblab/gcare: download `dataset.tar.gz`, `queryset.tar.gz` and `true_cardinality.tar.gz` (Google Drive links in its README) into `data/gcare/`; unpack the datasets in place, the queries into `queries/` and the true cardinalities into `truth/` (they share file names). `scripts/gcare_prepare.py` loads AIDS and YAGO into `gcare.duckdb`. |
| `imdb_cycles/` | IMDb cycle suite (`evaluate_imdb_suite.py`) | **Derived, not downloaded**: `scripts/prepare_imdb_cycles.py` projects JOB's full IMDb snapshot (`cast_info` by role, `movie_companies` by company type, `movie_keyword`) onto the 200 most-voted feature films (an earlier 2005 slice was too sparse; see the script docstring). Put the JOB CSVs in `data/job_full/` or point `LPH_JOB_DIR` at them; `MANIFEST.json` pins each output file's sha256, and evaluation refuses to run on a mismatch. The superseded 2005 pilot is `--pilot-year 2005 --n-movies 3500` (its 12 relation sha256s are in `results/imdb/superseded_sparse_2005/data_manifest.json`). |

The CorrBound comparison additionally reads the public MIT-licensed artifact
at https://github.com/christophmayerm/CorrBound (inspected revision
`8e05eb7`). Set `LPH_CORRBOUND_DIR` to that checkout; see
`scripts/corrbound_sm_artifact_lp.py` for the expected DBLP query/result
paths and `LPH_CORRBOUND_PYTHON` for its separate LP environment. The source
license does not establish redistribution rights for every bundled dataset.

## Checksums of the external downloads used for the paper

`sha256sum -c` from `data/` (the G-CARE archives as downloaded; the two
graph files after unpacking):

```
78b86cda06115c4554cdfcfb93a7fbc8ecb759df39927510dd02ced4228a95e4  gcare/dataset.tar.gz
c8dc9f978296559e9e55335a989ce16e7b5bcba7aa9d43e25fbd9e588d00ebc7  gcare/queryset.tar.gz
22e59f4fc06ffb79711d582513c6422ca555422c9947fde64a52f0a9292d382c  gcare/true_cardinality.tar.gz
379ef9537ac802437295adc890bf5ed1a502c5314627047583ff64458f094be7  gcare/aids/aids.txt
89a8a6e5d5fa52fa84ab058187bfc1a606485ac905ac4a2feb842e9a6fba68a2  gcare/yago/yago.txt
69a2dab71fa5e3a0715487599fc16ca17ddc847379325a6c765bbad6e3e36938  epinions/soc-Epinions1.txt.gz
```

The G-CARE cyclic query list is pinned separately
(`benchmarks/gcare_cyclic.json` and its `.sha256`).

## File format

Each relation is a CSV/TSV the loader reads as `(schema, rows)`; see the
`load_*` helpers in the consuming script for the exact column mapping
(e.g. `job_experiment.py:load_csv_relation`). Edge relations are expected
as two-column `(src, dst)` integer/string vertex identifiers.

> Note: except for the Hetionet edge lists shipped in `hetionet_cc0/`
> (upstream original content CC0; underlying source terms also apply), these datasets are external and not redistributed here. The pinned
> environment is `requirements.lock`; reproduction is
> `scripts/reproduce_all.sh`.
