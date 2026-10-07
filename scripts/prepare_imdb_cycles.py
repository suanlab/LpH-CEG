"""Derive the IMDb relations of the second real-data suite from JOB's IMDb.

Source: the IMDb snapshot of the Join Order Benchmark (Leis et al., VLDB
2015), the CSVs under ``JOB_DIR``.  We read four tables and project them
into *typed binary relations*, the analogue of Hetionet's metaedges:

* ``cast_info``  split by ``role_id``          -> ci_<role>  (Person, Movie)
* ``movie_companies`` split by ``company_type`` -> mc_<type>  (Movie, Company)
* ``movie_keyword``                             -> mk         (Movie, Keyword)

Scope rule: the ``N_MOVIES`` feature films (``kind_id = 1``) with the most
IMDb votes (``movie_info_idx``, info type 100; ties by ``title.id``).  Every
relation is restricted to that one movie set, so joins on the movie
variable see the same movies in every relation (a per-relation row cap, as
on Hetionet, would give each relation a different movie range), and
``N_MOVIES`` keeps the largest relation near the 20K rows per relation of
the Hetionet suites.

A first scope -- the first 3,500 feature films of 2005 by ``title.id`` --
was evaluated and replaced: its people rarely recur across films, so 82 of
the 120 registered cycles had an empty join and validity could not be
tested.  That run is kept in ``results/imdb/superseded_sparse_2005/``.  Duplicate (x, y) pairs are
removed: a relation is a set of edges.

Output: ``data/imdb_cycles/<relation>.tsv`` (header ``source<TAB>target``,
ids prefixed P/M/C/K) and ``data/imdb_cycles/MANIFEST.json``.

Usage:
    PYTHONPATH=. python scripts/prepare_imdb_cycles.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
# JOB's IMDb CSVs (https://github.com/gregrahn/join-order-benchmark);
# override with LPH_JOB_DIR.
JOB_DIR = Path(os.environ.get("LPH_JOB_DIR", str(ROOT / "data" / "job_full")))
OUT = ROOT / "data" / "imdb_cycles"
N_MOVIES = 200

ROLES = {  # role_id -> relation name (role_type.csv)
    1: "ci_actor",
    2: "ci_actress",
    3: "ci_producer",
    4: "ci_writer",
    5: "ci_cinematog",
    6: "ci_composer",
    8: "ci_director",
    9: "ci_editor",
    10: "ci_misccrew",
}
COMPANY_TYPES = {1: "mc_distrib", 2: "mc_produc"}  # company_type.csv

#: relation -> (source type, target type); the registry types cycles by these.
#: Each relation has its own non-movie entity type (people *as* actors, *as*
#: directors, ...; companies *as* distributors, ...), as each Hetionet
#: metaedge joins its own pair of node types.  A non-movie variable on a
#: cycle is then reached through one relation, used on both sides: two
#: films sharing an actor, a director, a distributor, a keyword.
IMDB_RELATIONS: dict[str, tuple[str, str]] = {
    **{name: (name.removeprefix("ci_"), "M") for name in ROLES.values()},
    **{name: ("M", name.removeprefix("mc_")) for name in COMPANY_TYPES.values()},
    "mk": ("M", "keyword"),
}


def _csv(
    con: duckdb.DuckDBPyConnection, table: str, cols: list[tuple[str, str]]
) -> None:
    spec = ", ".join(f"'{c}': '{t}'" for c, t in cols)
    con.execute(
        f"CREATE VIEW {table} AS SELECT * FROM read_csv('{JOB_DIR / table}.csv', "
        f"header=false, columns={{{spec}}}, quote='\"', escape='\\', "
        "ignore_errors=true)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-movies", type=int, default=N_MOVIES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    prepare(args.n_movies, args.out)


def prepare(n_movies: int, out_dir: Path) -> None:
    """Write the typed relations of the n_movies most-voted films to out_dir."""
    con = duckdb.connect()
    _csv(
        con,
        "title",
        [
            ("id", "BIGINT"),
            ("title", "VARCHAR"),
            ("imdb_index", "VARCHAR"),
            ("kind_id", "INT"),
            ("production_year", "INT"),
            ("imdb_id", "VARCHAR"),
            ("phonetic_code", "VARCHAR"),
            ("episode_of_id", "BIGINT"),
            ("season_nr", "INT"),
            ("episode_nr", "INT"),
            ("series_years", "VARCHAR"),
            ("md5sum", "VARCHAR"),
        ],
    )
    _csv(
        con,
        "cast_info",
        [
            ("id", "BIGINT"),
            ("person_id", "BIGINT"),
            ("movie_id", "BIGINT"),
            ("person_role_id", "BIGINT"),
            ("note", "VARCHAR"),
            ("nr_order", "INT"),
            ("role_id", "INT"),
        ],
    )
    _csv(
        con,
        "movie_companies",
        [
            ("id", "BIGINT"),
            ("movie_id", "BIGINT"),
            ("company_id", "BIGINT"),
            ("company_type_id", "INT"),
            ("note", "VARCHAR"),
        ],
    )
    _csv(
        con,
        "movie_keyword",
        [("id", "BIGINT"), ("movie_id", "BIGINT"), ("keyword_id", "BIGINT")],
    )
    _csv(
        con,
        "movie_info_idx",
        [
            ("id", "BIGINT"),
            ("movie_id", "BIGINT"),
            ("info_type_id", "INT"),
            ("info", "VARCHAR"),
            ("note", "VARCHAR"),
        ],
    )
    con.execute(
        "CREATE TABLE m AS SELECT t.id FROM title t JOIN movie_info_idx v "
        "ON v.movie_id = t.id AND v.info_type_id = 100 WHERE t.kind_id = 1 "
        f"ORDER BY CAST(v.info AS BIGINT) DESC, t.id LIMIT {n_movies}"
    )
    queries: dict[str, str] = {}
    for rid, name in ROLES.items():
        queries[name] = (
            "SELECT DISTINCT 'P' || person_id, 'M' || movie_id FROM cast_info "
            f"JOIN m ON movie_id = m.id WHERE role_id = {rid}"
        )
    for tid, name in COMPANY_TYPES.items():
        queries[name] = (
            "SELECT DISTINCT 'M' || movie_id, 'C' || company_id FROM movie_companies "
            f"JOIN m ON movie_id = m.id WHERE company_type_id = {tid}"
        )
    queries["mk"] = (
        "SELECT DISTINCT 'M' || movie_id, 'K' || keyword_id FROM movie_keyword "
        "JOIN m ON movie_id = m.id"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": str(JOB_DIR),
        "scope": "top-voted feature films",
        "n_movies": con.execute("SELECT count(*) FROM m").fetchone()[0],  # type: ignore[index]
        "relations": {},
    }
    for name, sql in queries.items():
        rows = sorted(con.execute(sql).fetchall())
        text = "source\ttarget\n" + "".join(f"{a}\t{b}\n" for a, b in rows)
        (out_dir / f"{name}.tsv").write_text(text)
        manifest["relations"][name] = {  # type: ignore[index]
            "types": IMDB_RELATIONS[name],
            "rows": len(rows),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        print(f"{name:<14} {len(rows):>7} rows")
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
