"""Load the G-CARE AIDS and YAGO data graphs into one DuckDB file.

G-CARE (Park et al., SIGMOD 2020; https://github.com/yspark-dblab/gcare)
ships labelled, directed data graphs as text: ``v <id> <label> ...`` (zero
or more vertex labels) and ``e <src> <dst> <label>``.  This writes, per
dataset, ``<ds>_vlabel(vid, label)`` and ``<ds>_edges(src, dst, label)`` to
``data/gcare/gcare.duckdb`` so that each query's relations can be cut out
with SQL.  Download and unpack G-CARE's ``dataset.tar.gz``,
``queryset.tar.gz`` and ``true_cardinality.tar.gz`` under ``data/gcare/``
(queries in ``queries/``, true counts in ``truth/``; see data/README.md).

Usage:
    PYTHONPATH=. python scripts/gcare_prepare.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
GCARE = Path(os.environ.get("LPH_GCARE_DIR", str(ROOT / "data" / "gcare")))
DATASETS = ("aids", "yago")


def _sql_str(path: object) -> str:
    """Escape a path for use inside a single-quoted SQL literal."""
    return str(path).replace("'", "''")


def main() -> None:
    con = duckdb.connect(str(GCARE / "gcare.duckdb"))
    for ds in DATASETS:
        with tempfile.TemporaryDirectory() as tmp:
            vpath, epath = Path(tmp) / "v.tsv", Path(tmp) / "e.tsv"
            with (
                open(GCARE / ds / f"{ds}.txt") as src,
                open(vpath, "w") as vf,
                open(epath, "w") as ef,
            ):
                for line in src:
                    t = line.split()
                    if not t:
                        continue
                    if t[0] == "v":
                        for lab in t[2:]:
                            vf.write(f"{t[1]}\t{lab}\n")
                    elif t[0] == "e":
                        ef.write(f"{t[1]}\t{t[2]}\t{t[3]}\n")
            con.execute(
                f"CREATE OR REPLACE TABLE {ds}_vlabel AS SELECT * FROM read_csv('{_sql_str(vpath)}', "
                "delim='\t', header=false, columns={'vid': 'BIGINT', 'label': 'INT'})"
            )
            con.execute(
                f"CREATE OR REPLACE TABLE {ds}_edges AS SELECT * FROM read_csv('{_sql_str(epath)}', "
                "delim='\t', header=false, columns={'src': 'BIGINT', 'dst': 'BIGINT', 'label': 'INT'})"
            )
        n_v = con.execute(f"SELECT count(*) FROM {ds}_vlabel").fetchone()[0]  # type: ignore[index]
        n_e = con.execute(f"SELECT count(*) FROM {ds}_edges").fetchone()[0]  # type: ignore[index]
        print(f"{ds}: {n_v} vertex labels, {n_e} edges", flush=True)
    con.close()


if __name__ == "__main__":
    main()
