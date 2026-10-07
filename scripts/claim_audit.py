"""Audit: every headline number in the paper must match its source CSV.

The paper's quantitative claims are spread across ~10 result CSVs.  Two
real defects have already been caught by tracing them by hand (a stale
`tab:scalability`, and an "LP overhead <3ms" claim contradicted by the
k=7 measurement).  This script turns that trace into a repeatable gate:
each entry below pins a claim as it appears in `main.tex` to the CSV cell
that must produce it.

A claim fails if the recomputed value no longer rounds to what the paper
prints.  Run it after regenerating any results CSV, before submitting.

Output: non-zero exit code if any claim drifted.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rel: str) -> list[dict[str, str]]:
    with open(ROOT / rel) as fh:
        return list(csv.DictReader(fh))


def pick(rows: list[dict[str, str]], col: str, **where: str) -> float:
    """Return `col` of the single row matching all `where` constraints."""
    hits = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if len(hits) != 1:
        raise LookupError(f"{where} matched {len(hits)} rows, expected 1")
    return float(hits[0][col])


def check(
    claim: str, actual: float, printed: float, tol: float = 0.005
) -> tuple[bool, str]:
    """Claim holds if `actual` rounds to `printed` within a relative tol."""
    ok = abs(actual - printed) <= tol * max(abs(printed), 1e-9)
    return ok, f"{claim:<52} paper={printed:<12g} csv={actual:<12.4f}"


def main() -> None:
    failures: list[str] = []
    lines: list[str] = []

    ms = load("results/revision/multi_seed.csv")
    sc = load("results/scalability/lp_scalability.csv")
    se = load("results/scalability/scale_entropic.csv")
    cr = load("results/scalability/crossover_real.csv")
    real = load("results/real/cyclic_real.csv")
    pj = load("results/e2e/plan_injection.csv")
    rb = load("results/e2e/robustness_broad_summary.csv")
    fit = load("results/characterization/skew_gap_fit.csv")
    ordr = load("results/wcoj/ordering_study_summary.csv")
    pv = load("results/ablation/pvalue_ablation.csv")
    sub = load("results/ablation/substitutes_ablation.csv")
    val = load("results/validity/validity_search.csv")
    suite = load("results/suite/suite_summary.csv")
    cost = load("results/cost/stats_cost.csv")
    shannon = load("results/validity/shannon_implication.csv")
    wide = load("results/validity/shannon_wide.csv")
    svi = load("results/ablation/statistics_vs_inference.csv")
    rfr = load("results/ablation/real_frontier.csv")
    hb2 = load("results/revision/headline_bs2.csv")
    lsu = load("results/suite/long_suite_summary.csv")
    lse = load("results/suite/long_suite_eval.csv")
    apj = load("results/suite/agm_pairjoin_summary.csv")
    apje = load("results/suite/agm_pairjoin.csv")
    cat = load("results/suite/stats_catalogue.csv")
    cats = load("results/suite/stats_catalogue_summary.csv")
    f22 = load("results/flow/flow_2x2.csv")
    fsc = load("results/flow/flow_scale.csv")
    fva = load("results/flow/flow_validity_summary.csv")[0]
    fls = load("results/flow/flow_long_suite_summary.csv")
    pls = load("results/pushdown/pushdown_long_suite_summary.csv")
    bcmp = load("results/baselines/baseline_comparison.csv")
    sent = load("results/scalability/scale_entropic.csv")
    fl_rows = load("results/flow/flow_long_suite.csv")
    cbp = load("results/pushdown/corrbound_proxy_summary.csv")
    rft = load("results/ablation/real_frontier_truth.csv")
    ims = load("results/imdb/imdb_suite_summary.csv")
    isc = load("results/imdb/imdb_scale_n1000_summary.csv")
    ipi = load("results/imdb/imdb_plan_injection_summary.csv")
    ime = load("results/imdb/imdb_suite_eval.csv")
    imsp = load("results/imdb/superseded_sparse_2005/imdb_suite_eval.csv")
    lss = load("results/suite/long_suite_summary.csv")
    imreg = json.loads((ROOT / "benchmarks" / "query_suite_v3_imdb.json").read_text())

    def het_flow(nv: str, num: str) -> float:
        rs = [r for r in fl_rows if r["n_vars"] == nv]
        if num == "true":
            return statistics.median(
                float(r["flow_bs2"]) / int(r["true_count"])
                for r in rs
                if r["true_count"] not in ("", "0")
            )
        return statistics.median(
            float(r["flow_bs2"]) / float(r["entropic_bs1"]) for r in rs
        )

    p22 = load("results/pushdown/pushdown_2x2.csv")
    psc = load("results/pushdown/pushdown_scale.csv")

    def cyc(q: str) -> float:
        return pick(ms, "ratio_mean", experiment="cycle_scale", query=q)

    checks: list[tuple] = [
        # -- tab:synthetic / abstract headline ------------------------------
        ("cycle-8 headline ratio (614x)", cyc("cycle_8"), 613.7641),
        ("cycle-7 ratio (211x)", cyc("cycle_7"), 211.3454),
        ("cycle-6 ratio (69x)", cyc("cycle_6"), 68.9005),
        ("cycle-3 ratio (2.8x, range floor)", cyc("cycle_3"), 2.7486),
        # -- tab:4way data scaling -----------------------------------------
        (
            "data scale 100V (5.9x)",
            pick(ms, "ratio_mean", experiment="data_scale", n_vertices="100"),
            5.9342,
        ),
        (
            "data scale 2KV (33.9x)",
            pick(ms, "ratio_mean", experiment="data_scale", n_vertices="2000"),
            33.9144,
        ),
        # -- tab:skew (peak at alpha=2) ------------------------------------
        (
            "skew peak alpha=2 (16.1x)",
            pick(ms, "ratio_mean", experiment="skew_a=2.0"),
            16.0773,
        ),
        # -- acyclic control: LP must give exactly no improvement ----------
        (
            "acyclic star_3 = 1.00x (no LP gain)",
            pick(ms, "ratio_mean", experiment="acyclic", query="star_3"),
            1.0,
        ),
        # -- tab:scalability / LP overhead ---------------------------------
        ("cycle-10 improvement (971x)", pick(sc, "improvement", k="10"), 970.6752),
        # -- entropic crossover (C1) ---------------------------------------
        ("cover LP solve at k=12 (33ms)", pick(se, "cover_ms", k="12"), 33.0980),
        (
            "entropic LP at k=12 (43s, interior-point)",
            pick(se, "entropic_ms", k="12") / 1000.0,
            42.9301,
        ),
        (
            "entropic LP at k=14 (3,679s -- NOT intractable)",
            pick(se, "entropic_ms", k="14") / 1000.0,
            3679.3821,
        ),
        (
            "max synthetic cover/entropic gap (1.43x)",
            max(float(r["gap_cover_over_entropic"]) for r in se if r["status"] == "ok"),
            1.4293,
        ),
        (
            "real crossover k=12 cover (1.68s)",
            pick(cr, "cover_ms", k="12") / 1000.0,
            1.6752,
        ),
        (
            "real crossover k=12 entropic (46s, interior-point)",
            pick(cr, "entropic_ms", k="12") / 1000.0,
            45.6589,
        ),
        (
            "real crossover k=14 entropic (2,319s -- solves)",
            pick(cr, "entropic_ms", k="14") / 1000.0,
            2318.9376,
        ),
        (
            "extreme real gap (8.8x)",
            max(float(r["gap_cover_over_entropic"]) for r in cr if r["status"] == "ok"),
            8.7511,
        ),
        # -- real data (Hetionet) ------------------------------------------
        (
            "Hetionet cycle-4 improvement (7.59x)",
            pick(real, "improvement", cycle="C-D-G-G cycle-4"),
            7.5871,
        ),
        # -- tab:robust tail (C4) ------------------------------------------
        (
            "bound-guided worst slowdown (3.6x)",
            max(float(r["bound_guided_slowdown"]) for r in pj),
            3.5647,
        ),
        (
            "DuckDB worst slowdown (21.0x)",
            max(float(r["duckdb_slowdown"]) for r in pj),
            21.0496,
        ),
        # -- broad tail distribution (C4) ----------------------------------
        (
            "bound-guided P95 (1.21x, AGM-censored)",
            pick(rb, "p95", strategy="bound-guided"),
            1.210,
        ),
        ("DuckDB P95 (5.82x)", pick(rb, "p95", strategy="DuckDB-opt"), 5.816),
        # -- gap predictor (C2) --------------------------------------------
        ("held-out R^2 (0.87)", pick(fit, "value", metric="r2_heldout"), 0.8738),
        (
            "Spearman rho (-0.74)",
            pick(fit, "value", metric="spearman_maxskew_gap"),
            -0.735,
        ),
        # -- WCOJ ordering honest negative (C3) ----------------------------
        (
            "LpH vs H-CEG median ratio (-1.3%)",
            pick(ordr, "median_ratio", comparison="LpH vs H-CEG"),
            0.988,
        ),
        (
            "LpH vs H-CEG Cliff's delta (-0.35)",
            pick(ordr, "cliffs_delta", comparison="LpH vs H-CEG"),
            -0.348,
        ),
        # -- ablations ------------------------------------------------------
        (
            "p beyond {1,2,inf} adds <2.4%",
            max(
                (float(r["{1,1.5,2,3,5,inf}"]) / float(r["{1,2,inf}"]) - 1) * 100
                for r in pv
            ),
            2.3615,
        ),
        (
            "LP complementary on cycle-6 (1.57x)",
            pick(sub, "block2_over_lp2", family="cycle6"),
            1.5681,
        ),
        (
            "LP complementary on K4 (1.20x)",
            pick(sub, "block2_over_lp2", family="k4"),
            1.1996,
        ),
        # -- pre-registered public suite (W6) -------------------------------
        (
            "suite gap median (1.99x, hetionet)",
            pick(suite, "gap_median", scope="hetionet"),
            1.9853,
        ),
        ("suite gap P90 (3.55x)", pick(suite, "gap_p90", scope="hetionet"), 3.5468),
        ("suite gap max (6.29x)", pick(suite, "gap_max", scope="hetionet"), 6.2931),
        (
            "suite improvement median (1.47x)",
            pick(suite, "improvement_median", scope="hetionet"),
            1.4657,
        ),
        (
            "suite queries censored (0)",
            pick(suite, "n_censored", scope="ALL"),
            0,
        ),
        (
            "suite registered size (377)",
            pick(suite, "n_registered", scope="ALL"),
            377,
        ),
        (
            "epinions is degenerate (1 distinct gap)",
            pick(suite, "n_distinct_gaps", scope="epinions"),
            1,
        ),
        # -- total estimation cost ------------------------------------------
        (
            "LP share of total cost at bs=1 (~7%)",
            statistics.median(
                float(r["lp_share_of_total"]) for r in cost if r["block_size"] == "1"
            ),
            0.0727,
        ),
        # -- Step-5 obligation over the Shannon cone ------------------------
        (
            "Shannon-cone checks all certified",
            sum(1 for r in shannon if r["shannon_provable"] == "True"),
            len(shannon),
        ),
        (
            "Shannon-cone wide check: every checked hyperpath holds",
            sum(1 for r in wide if r["all_hold"] == "True"),
            len(wide),
        ),
        (
            "Shannon-cone wide check: (shape, bs) pairs complete",
            sum(1 for r in wide if r["complete"] == "True"),
            19,
        ),
        # -- statistics vs inference (the paper's central claim) ------------
        (
            "cover/block beats entropic/single on all pairs",
            sum(1 for r in svi if float(r["cover_block_vs_entropic_single"]) < 1.0),
            len(svi),
        ),
        (
            "entropic wins at EQUAL statistics on all pairs",
            sum(1 for r in svi if r["entropic_wins_at_equal_stats"] == "True"),
            len(svi),
        ),
        (
            "real suite: bs=2 tighter than entropic on all 109",
            sum(1 for r in rfr if r["bs2_beats_entropic"] == "True"),
            len(rfr),
        ),
        (
            "real suite median bs2/entropic (0.057x)",
            statistics.median(float(r["bs2_over_entropic"]) for r in rfr),
            0.0570,
            0.02,
        ),
        # -- three-axis comparison (Section: statistics vs inference) -------
        (
            "inference gain max-deg->LP at k=8 (654x)",
            pick(hb2, "inference_gain", k="8"),
            654.2,
            0.01,
        ),
        (
            "statistics gain bs1->bs2 at k=8 (2.70x)",
            pick(hb2, "statistics_gain", k="8"),
            2.7040,
            0.01,
        ),
        (
            "statistics gain bs1->bs2 at k=4 (1.21x)",
            pick(hb2, "statistics_gain", k="4"),
            1.2141,
            0.01,
        ),
        # -- long-cycle suite, |V| in {6,8,10} (tab:long-suite) -------------
        (
            "long suite bs1/entropic median at |V|=10 (3.61x)",
            pick(lsu, "bs1_over_entropic_median", n_vars="10"),
            3.6089,
        ),
        (
            "long suite bs2/entropic median at |V|=10 (0.00039x)",
            pick(lsu, "bs2_over_entropic_median", n_vars="10"),
            0.000392,
            0.01,
        ),
        (
            "long suite bs2 below entropic on all 120",
            sum(1 for r in lse if r["bs2_beats_entropic"] == "True"),
            120,
        ),
        (
            "long suite validity failures (0 of 345 checks)",
            sum(
                1
                for r in lse
                for c in ("valid_bs1", "valid_bs2", "valid_entropic")
                if r[c] == "False"
            ),
            0,
        ),
        (
            "long suite validity checks performed (345)",
            sum(
                1
                for r in lse
                for c in ("valid_bs1", "valid_bs2", "valid_entropic")
                if r[c] in ("True", "False")
            ),
            345,
        ),
        (
            "long suite bs2 timeouts flagged (33)",
            sum(1 for r in lse if "bs2_timeout" in r["censored"]),
            33,
        ),
        # -- AGM + pair-join sizes: the "known trick" baseline --------------
        (
            "AGM+pairjoins/entropic median |V|=6 (0.067)",
            pick(apj, "median_agm_pairjoins_over_entropic", n_vars="6"),
            0.067,
            0.01,
        ),
        (
            "AGM+pairjoins/entropic median |V|=10 (0.085)",
            pick(apj, "median_agm_pairjoins_over_entropic", n_vars="10"),
            0.085,
            0.01,
        ),
        (
            "bs2/AGM+pairjoins median |V|=6 (0.22)",
            pick(apj, "median_bs2_over_agm_pairjoins", n_vars="6"),
            0.22,
            0.01,
        ),
        (
            "bs2/AGM+pairjoins median |V|=10 (0.0047)",
            pick(apj, "median_bs2_over_agm_pairjoins", n_vars="10"),
            0.0047,
            0.02,
        ),
        (
            "bs2 below AGM+pairjoins on 120/120",
            sum(1 for r in apje if float(r["bs2_over_agm_pairjoins"]) < 1.0),
            len(apje),
        ),
        # -- B2: amortised statistics catalogue -----------------------------
        (
            "catalogue entries (676)",
            max(int(r["catalogue_size"]) for r in cat),
            676,
        ),
        ("catalogue block builds (91)", sum(int(r["misses"]) for r in cat), 91),
        (
            "catalogue bound == per-query bound on 120/120",
            sum(1 for r in cat if r["bound_equal"] == "True"),
            len(cat),
        ),
        (
            "catalogue build, whole workload (~9 min = 527 s)",
            sum(float(r["stats_s"]) for r in cat),
            527,
            0.25,
        ),
        (
            "per-query statistics, whole workload (~3.3 h = 11,765 s)",
            sum(float(r["cold_bs2_s"]) for r in cat),
            11765,
            0.05,
        ),
        (
            "warm per-query cost |V|=10 median (~0.010 s)",
            pick(cats, "median_total_s", subset="all", n_vars="10"),
            0.010,
            0.5,
        ),
        (
            "warm per-query cost |V|=8 median (~0.007 s)",
            pick(cats, "median_total_s", subset="all", n_vars="8"),
            0.007,
            0.5,
        ),
        (
            "second half below entropic on 60/60",
            pick(cats, "n_total_below_entropic", subset="second_half", n_vars="all"),
            60,
        ),
        # -- B3 flow-LP inference over block statistics ---------------------
        (
            "flow LP == entropic at bs=1 on 12/12 instances",
            sum(1 for r in f22 if r["flow_bs1_eq_entropic"] == "True"),
            12,
        ),
        (
            "flow LP == 2^|V| polymatroid at bs=2 on 12/12 instances",
            sum(1 for r in f22 if float(r["flow_bs2_over_poly_bs2"]) < 1 + 1e-6),
            12,
        ),
        (
            "cover bs=2 / flow bs=2, small families, min (1.5x)",
            min(float(r["cover_bs2_over_flow_bs2"]) for r in f22),
            1.5,
            0.03,
        ),
        (
            "cover bs=2 / flow bs=2, small families, max (3.8x)",
            max(float(r["cover_bs2_over_flow_bs2"]) for r in f22),
            3.8,
            0.02,
        ),
        (
            "dropping |X|=2 constraints loosens polymatroid, min (1.5x)",
            min(float(r["poly_bs2_simple"]) / float(r["poly_bs2"]) for r in f22),
            1.5,
            0.03,
        ),
        (
            "dropping |X|=2 constraints loosens polymatroid, max (2.9x)",
            max(float(r["poly_bs2_simple"]) / float(r["poly_bs2"]) for r in f22),
            2.9,
            0.01,
        ),
        (
            "flow LP == entropic at bs=1 for every k<=14 on the sweep (9)",
            sum(1 for r in fsc if r["flow_bs1_eq_entropic"] == "True"),
            9,
        ),
        (
            "flow LP bs=1 solve, k=3..16, min (3 ms)",
            min(float(r["flow_bs1_ms"]) for r in fsc),
            3,
            0.1,
        ),
        (
            "flow LP bs=1 solve, k=3..16, max (26 ms)",
            max(float(r["flow_bs1_ms"]) for r in fsc),
            26,
            0.02,
        ),
        (
            "flow LP bs=2 over 4k orders, total, min (0.03 s)",
            min(float(r["flow_bs2_ms_total"]) for r in fsc) / 1000.0,
            0.03,
            0.1,
        ),
        (
            "flow LP bs=2 over 4k orders, total, max (4.4 s)",
            max(float(r["flow_bs2_ms_total"]) for r in fsc) / 1000.0,
            4.4,
            0.02,
        ),
        (
            "flow LP bs=2 per order, max (73 ms)",
            max(float(r["flow_bs2_ms_per_order"]) for r in fsc),
            73,
            0.01,
        ),
        (
            "flow bs=2 == polymatroid on every even k<=10 (4 of 4)",
            sum(
                1
                for r in fsc
                if int(r["k"]) % 2 == 0
                and int(r["k"]) <= 10
                and float(r["flow_bs2_over_poly_bs2"]) < 1 + 1e-6
            ),
            4,
        ),
        (
            "flow bs=2 / polymatroid on odd k, min (1.56x)",
            min(
                float(r["flow_bs2_over_poly_bs2"]) for r in fsc if int(r["k"]) in (5, 7)
            ),
            1.56,
        ),
        (
            "flow bs=2 / polymatroid on odd k, max (1.59x)",
            max(
                float(r["flow_bs2_over_poly_bs2"]) for r in fsc if int(r["k"]) in (5, 7)
            ),
            1.59,
        ),
        (
            "2^|V| polymatroid at bs=2, k=10 (0.7 s)",
            pick(fsc, "poly_bs2_ms", k="10") / 1000.0,
            0.7,
            0.03,
        ),
        (
            "cover bs=2 / flow bs=2 at k=3 (2.2x)",
            pick(fsc, "cover_bs2_over_flow_bs2", k="3"),
            2.2,
        ),
        (
            "cover bs=2 / flow bs=2 at k=16 (288x)",
            pick(fsc, "cover_bs2_over_flow_bs2", k="16"),
            288,
        ),
        (
            "bs=1 optimum / flow bs=2, max over sweep (800x)",
            max(
                float(r["entropic_bound"]) / float(r["flow_bs2_bound"])
                for r in fsc
                if r["entropic_bound"]
            ),
            800,
            0.05,
        ),
        (
            "bs=1 optimum / flow bs=2, min over sweep (2x)",
            min(
                float(r["entropic_bound"]) / float(r["flow_bs2_bound"])
                for r in fsc
                if r["entropic_bound"]
            ),
            2,
            0.15,
        ),
        (
            "bs=1 optimum / cover bs=2, max over sweep (6x)",
            max(
                float(r["entropic_bound"]) / float(r["cover_bs2_bound"])
                for r in fsc
                if r["entropic_bound"]
            ),
            6,
            0.05,
        ),
        ("flow validity: random shapes (40)", float(fva["n_instances"]), 40),
        (
            "flow validity: order-instance pairs (2,046)",
            float(fva["n_order_pairs"]),
            2046,
        ),
        (
            "flow validity: never below relaxed polymatroid (0)",
            float(fva["worst_flow_below_relaxed"]),
            0,
            1.0,  # absolute 1e-9: check() scales tol by max(|printed|, 1e-9)
        ),
        (
            "flow validity: valid vs exact join on 40/40",
            float(fva["n_valid_vs_exact"]),
            40,
        ),
        (
            "flow validity: exact polymatroid reached on 34/40",
            float(fva["n_flow_reaches_poly_all"]),
            34,
        ),
        (
            "flow validity: max gap to polymatroid (1.28x)",
            2 ** float(fva["max_gap_bits"]),
            1.28,
        ),
        (
            "flow long suite: median cover/flow at |V|=6 (3.2x)",
            pick(fls, "median_cover_bs2_over_flow_bs2", n_vars="6"),
            3.2,
            0.02,
        ),
        (
            "flow long suite: median cover/flow at |V|=8 (4.2x)",
            pick(fls, "median_cover_bs2_over_flow_bs2", n_vars="8"),
            4.2,
            0.02,
        ),
        (
            "flow long suite: median cover/flow at |V|=10 (5.1x)",
            pick(fls, "median_cover_bs2_over_flow_bs2", n_vars="10"),
            5.1,
            0.02,
        ),
        (
            "flow long suite: flow == polymatroid on 104/120",
            pick(fls, "n_flow_bs2_eq_poly_bs2", n_vars="all"),
            104,
        ),
        (
            "flow long suite: max gap to polymatroid (1.73x)",
            pick(fls, "max_flow_bs2_over_poly_bs2", n_vars="all"),
            1.73,
        ),
        (
            "flow long suite: valid on all 115 checked",
            pick(fls, "n_valid_flow_bs2", n_vars="all"),
            115,
        ),
        (
            "flow long suite: n_checked (115)",
            pick(fls, "n_checked", n_vars="all"),
            115,
        ),
        (
            "flow long suite: median flow bs=2 s at |V|=6 (0.27 s)",
            pick(fls, "median_flow_bs2_s_total", n_vars="6"),
            0.27,
            0.02,
        ),
        (
            "flow long suite: median flow bs=2 s at |V|=10 (1.1 s)",
            pick(fls, "median_flow_bs2_s_total", n_vars="10"),
            1.1,
            0.02,
        ),
        (
            "flow long suite: median entropic s at |V|=6 (0.5 s)",
            pick(fls, "median_entropic_s", n_vars="6"),
            0.5,
            0.06,
        ),
        (
            "flow long suite: median entropic s at |V|=10 (2.2 s)",
            pick(fls, "median_entropic_s", n_vars="10"),
            2.2,
            0.02,
        ),
        # -- 2x2 on one identical catalogue (flow_2x2.csv) -----------------
        (
            "2x2 table: triangle optimum on blocks (368)",
            pick(f22, "poly_bs2", family="triangle", seed="101"),
            368,
            0.002,
        ),
        (
            "2x2 table: cycle-4 optimum on blocks (1,355)",
            pick(f22, "poly_bs2", family="cycle4", seed="101"),
            1355,
            0.001,
        ),
        (
            "2x2 table: diamond optimum on blocks (1,355)",
            pick(f22, "poly_bs2", family="diamond", seed="101"),
            1355,
            0.001,
        ),
        (
            "2x2: stats axis, cover LP, min (1.5x)",
            min([float(r["cover_bs1"]) / float(r["cover_bs2"]) for r in f22]),
            1.5,
            0.02,
        ),
        (
            "2x2: stats axis, cover LP, max (2.3x)",
            max([float(r["cover_bs1"]) / float(r["cover_bs2"]) for r in f22]),
            2.3,
            0.01,
        ),
        (
            "2x2: stats axis, optimum, min (1.7x)",
            min([float(r["entropic_bs1"]) / float(r["poly_bs2"]) for r in f22]),
            1.7,
            0.02,
        ),
        (
            "2x2: stats axis, optimum, max (5.5x)",
            max([float(r["entropic_bs1"]) / float(r["poly_bs2"]) for r in f22]),
            5.5,
            0.01,
        ),
        (
            "2x2: inference axis, single, min (1.15x)",
            min([float(r["cover_bs1"]) / float(r["entropic_bs1"]) for r in f22]),
            1.15,
            0.005,
        ),
        (
            "2x2: inference axis, single, max (1.5x)",
            max([float(r["cover_bs1"]) / float(r["entropic_bs1"]) for r in f22]),
            1.5,
            0.01,
        ),
        (
            "2x2: inference axis, block, min (1.5x)",
            min([float(r["cover_bs2"]) / float(r["poly_bs2"]) for r in f22]),
            1.5,
            0.034,
        ),
        (
            "2x2: inference axis, block, max (3.8x)",
            max([float(r["cover_bs2"]) / float(r["poly_bs2"]) for r in f22]),
            3.8,
            0.014,
        ),
        (
            "2x2: flow bs2 == polymatroid on blocks (12/12)",
            sum(abs(float(r["flow_bs2_over_poly_bs2"]) - 1) < 1e-3 for r in f22),
            12,
        ),
        (
            "sweep: flow over cover on bs2, cycle-3 (2.2x)",
            pick(fsc, "cover_bs2_over_flow_bs2", k="3"),
            2.2,
            0.01,
        ),
        (
            "sweep: flow over cover on bs2, cycle-4 (7.1x)",
            pick(fsc, "cover_bs2_over_flow_bs2", k="4"),
            7.1,
            0.01,
        ),
        (
            "real cycles: cover/entropic min (1.9x)",
            min(
                float(r["cover_vs_entropic"]) for r in bcmp if "Hetionet" in r["query"]
            ),
            1.9,
            0.01,
        ),
        (
            "real cycles: cover/entropic max (3.5x)",
            max(
                float(r["cover_vs_entropic"]) for r in bcmp if "Hetionet" in r["query"]
            ),
            3.5,
            0.015,
        ),
        (
            "entropic 2^|V| LP at |V|=8 (111 ms)",
            pick(sent, "entropic_ms", k="8"),
            111,
            0.005,
        ),
        (
            "long suite: flow bs1 == entropic (120/120)",
            sum(r["flow_bs1_eq_entropic"] == "True" for r in fl_rows),
            120,
        ),
        (
            "pushdown: median bs1+l2 codegree/full on suite (1.03)",
            pick(pls, "median_bs1_l2join_over_full", n_vars="all"),
            1.03,
            0.005,
        ),
        (
            "pushdown: bs1+l2 codegree == full on suite (53/120)",
            pick(pls, "n_bs1_l2join_eq_full", n_vars="all"),
            53,
        ),
        # -- real frontier: absolute distance, and the 229 real queries ----
        (
            "real frontier: 20 tightest checked, all valid (20)",
            sum(r["valid"] == "True" for r in rft),
            20,
        ),
        (
            "real frontier: max true count on the 20 (70)",
            max(int(r["true_count"]) for r in rft),
            70,
        ),
        (
            "real frontier: min bs2 on the 20 (291)",
            min(float(r["bs2"]) for r in rft),
            291,
        ),
        (
            "real frontier: max bs2 on the 20 (827)",
            max(float(r["bs2"]) for r in rft),
            827,
        ),
        (
            "real queries below entropic at bs=2 (229 = 109 + 120)",
            sum(r["bs2_beats_entropic"] == "True" for r in rfr)
            + sum(
                r["bs2_beats_entropic"] == "True"
                for r in load("results/suite/long_suite_eval.csv")
            ),
            229,
        ),
        # -- IMDb suite (second real dataset) + Hetionet flow columns -----
        (
            "Hetionet long: maxdeg/cover bs1 |V|=6 (1.8x)",
            pick(lss, "improvement_median", n_vars="6"),
            1.8,
            0.03,
        ),
        (
            "Hetionet long: maxdeg/cover bs1 |V|=8 (5.1x)",
            pick(lss, "improvement_median", n_vars="8"),
            5.1,
            0.03,
        ),
        (
            "Hetionet long: maxdeg/cover bs1 |V|=10 (8.9x)",
            pick(lss, "improvement_median", n_vars="10"),
            8.9,
            0.03,
        ),
        (
            "Hetionet long: flow bs2/entropic |V|=6 (0.0035)",
            het_flow("6", "ent"),
            0.0035,
            0.03,
        ),
        ("Hetionet long: flow bs2/exact |V|=6 (81)", het_flow("6", "true"), 81, 0.01),
        (
            "Hetionet long: flow bs2/entropic |V|=8 (0.00038)",
            het_flow("8", "ent"),
            0.00038,
            0.03,
        ),
        ("Hetionet long: flow bs2/exact |V|=8 (406)", het_flow("8", "true"), 406, 0.01),
        (
            "Hetionet long: flow bs2/entropic |V|=10 (6.3e-05)",
            het_flow("10", "ent"),
            6.3e-05,
            0.03,
        ),
        (
            "Hetionet long: flow bs2/exact |V|=10 (5600)",
            het_flow("10", "true"),
            5600,
            0.01,
        ),
        (
            "IMDb: median_improvement_maxdeg_over_bs1 |V|=6 (1.8)",
            pick(ims, "median_improvement_maxdeg_over_bs1", n_vars="6"),
            1.8,
            0.03,
        ),
        (
            "IMDb: median_improvement_maxdeg_over_bs1 |V|=8 (1.9)",
            pick(ims, "median_improvement_maxdeg_over_bs1", n_vars="8"),
            1.9,
            0.03,
        ),
        (
            "IMDb: median_improvement_maxdeg_over_bs1 |V|=10 (3.2)",
            pick(ims, "median_improvement_maxdeg_over_bs1", n_vars="10"),
            3.2,
            0.02,
        ),
        (
            "IMDb: median_bs1_over_entropic |V|=6 (3.75)",
            pick(ims, "median_bs1_over_entropic", n_vars="6"),
            3.75,
            0.005,
        ),
        (
            "IMDb: median_bs1_over_entropic |V|=8 (4.33)",
            pick(ims, "median_bs1_over_entropic", n_vars="8"),
            4.33,
            0.005,
        ),
        (
            "IMDb: median_bs1_over_entropic |V|=10 (4.14)",
            pick(ims, "median_bs1_over_entropic", n_vars="10"),
            4.14,
            0.005,
        ),
        (
            "IMDb: median_bs2_over_entropic |V|=6 (0.25)",
            pick(ims, "median_bs2_over_entropic", n_vars="6"),
            0.25,
            0.02,
        ),
        (
            "IMDb: median_bs2_over_entropic |V|=8 (0.075)",
            pick(ims, "median_bs2_over_entropic", n_vars="8"),
            0.075,
            0.01,
        ),
        (
            "IMDb: median_bs2_over_entropic |V|=10 (0.023)",
            pick(ims, "median_bs2_over_entropic", n_vars="10"),
            0.023,
            0.01,
        ),
        (
            "IMDb: median_flow_bs2_over_entropic |V|=6 (0.06)",
            pick(ims, "median_flow_bs2_over_entropic", n_vars="6"),
            0.06,
            0.01,
        ),
        (
            "IMDb: median_flow_bs2_over_entropic |V|=8 (0.019)",
            pick(ims, "median_flow_bs2_over_entropic", n_vars="8"),
            0.019,
            0.03,
        ),
        (
            "IMDb: median_flow_bs2_over_entropic |V|=10 (0.0056)",
            pick(ims, "median_flow_bs2_over_entropic", n_vars="10"),
            0.0056,
            0.01,
        ),
        (
            "IMDb: n_bs2_beats_entropic |V|=6 (40)",
            pick(ims, "n_bs2_beats_entropic", n_vars="6"),
            40,
        ),
        (
            "IMDb: n_bs2_beats_entropic |V|=8 (40)",
            pick(ims, "n_bs2_beats_entropic", n_vars="8"),
            40,
        ),
        (
            "IMDb: n_bs2_beats_entropic |V|=10 (40)",
            pick(ims, "n_bs2_beats_entropic", n_vars="10"),
            40,
        ),
        (
            "IMDb: median_sizes_over_entropic |V|=6 (1.0)",
            pick(ims, "median_sizes_over_entropic", n_vars="6"),
            1.0,
        ),
        (
            "IMDb: median_sizes_over_entropic |V|=8 (1.0)",
            pick(ims, "median_sizes_over_entropic", n_vars="8"),
            1.0,
        ),
        (
            "IMDb: median_sizes_over_entropic |V|=10 (1.0)",
            pick(ims, "median_sizes_over_entropic", n_vars="10"),
            1.0,
        ),
        (
            "IMDb: median_cover_bs2_over_flow_bs2 |V|=6 (4.2)",
            pick(ims, "median_cover_bs2_over_flow_bs2", n_vars="6"),
            4.2,
            0.01,
        ),
        (
            "IMDb: median_cover_bs2_over_flow_bs2 |V|=8 (4.0)",
            pick(ims, "median_cover_bs2_over_flow_bs2", n_vars="8"),
            4.0,
            0.01,
        ),
        (
            "IMDb: median_cover_bs2_over_flow_bs2 |V|=10 (3.3)",
            pick(ims, "median_cover_bs2_over_flow_bs2", n_vars="10"),
            3.3,
            0.01,
        ),
        (
            "IMDb: n_flow_eq_poly |V|=6 (28)",
            pick(ims, "n_flow_eq_poly", n_vars="6"),
            28,
        ),
        (
            "IMDb: n_flow_eq_poly |V|=8 (27)",
            pick(ims, "n_flow_eq_poly", n_vars="8"),
            27,
        ),
        (
            "IMDb: n_flow_eq_poly |V|=10 (30)",
            pick(ims, "n_flow_eq_poly", n_vars="10"),
            30,
        ),
        (
            "IMDb: median_entropic_over_true |V|=6 (74)",
            pick(ims, "median_entropic_over_true", n_vars="6"),
            74,
            0.01,
        ),
        (
            "IMDb: median_entropic_over_true |V|=8 (970)",
            pick(ims, "median_entropic_over_true", n_vars="8"),
            970,
            0.005,
        ),
        (
            "IMDb: median_entropic_over_true |V|=10 (9200)",
            pick(ims, "median_entropic_over_true", n_vars="10"),
            9200,
            0.005,
        ),
        (
            "IMDb: median_flow_bs2_over_true |V|=6 (4.2)",
            pick(ims, "median_flow_bs2_over_true", n_vars="6"),
            4.2,
            0.01,
        ),
        (
            "IMDb: median_flow_bs2_over_true |V|=8 (16)",
            pick(ims, "median_flow_bs2_over_true", n_vars="8"),
            16,
            0.01,
        ),
        (
            "IMDb: median_flow_bs2_over_true |V|=10 (60)",
            pick(ims, "median_flow_bs2_over_true", n_vars="10"),
            60,
            0.01,
        ),
        ("IMDb: n_checked |V|=6 (40)", pick(ims, "n_checked", n_vars="6"), 40),
        ("IMDb: n_checked |V|=8 (33)", pick(ims, "n_checked", n_vars="8"), 33),
        ("IMDb: n_checked |V|=10 (28)", pick(ims, "n_checked", n_vars="10"), 28),
        ("IMDb: n_invalid |V|=6 (0)", pick(ims, "n_invalid", n_vars="6"), 0),
        ("IMDb: n_invalid |V|=8 (0)", pick(ims, "n_invalid", n_vars="8"), 0),
        ("IMDb: n_invalid |V|=10 (0)", pick(ims, "n_invalid", n_vars="10"), 0),
        (
            "IMDb: median_cover_bs1_s |V|=6 (0.3)",
            pick(ims, "median_cover_bs1_s", n_vars="6"),
            0.3,
            0.06,
        ),
        (
            "IMDb: median_cover_bs1_s |V|=8 (0.6)",
            pick(ims, "median_cover_bs1_s", n_vars="8"),
            0.6,
            0.06,
        ),
        (
            "IMDb: median_cover_bs1_s |V|=10 (0.9)",
            pick(ims, "median_cover_bs1_s", n_vars="10"),
            0.9,
            0.01,
        ),
        (
            "IMDb: median_entropic_s |V|=6 (0.3)",
            pick(ims, "median_entropic_s", n_vars="6"),
            0.3,
            0.15,
        ),
        (
            "IMDb: median_entropic_s |V|=8 (0.8)",
            pick(ims, "median_entropic_s", n_vars="8"),
            0.8,
            0.01,
        ),
        (
            "IMDb: median_entropic_s |V|=10 (3.3)",
            pick(ims, "median_entropic_s", n_vars="10"),
            3.3,
            0.02,
        ),
        (
            "IMDb: median_cover_bs2_s |V|=6 (4.4)",
            pick(ims, "median_cover_bs2_s", n_vars="6"),
            4.4,
            0.01,
        ),
        (
            "IMDb: median_cover_bs2_s |V|=8 (14)",
            pick(ims, "median_cover_bs2_s", n_vars="8"),
            14,
            0.01,
        ),
        (
            "IMDb: median_cover_bs2_s |V|=10 (22)",
            pick(ims, "median_cover_bs2_s", n_vars="10"),
            22,
            0.01,
        ),
        (
            "IMDb: flow bs2 over sizes, median all (49x)",
            pick(ims, "median_sizes_over_flow_bs2", n_vars="all"),
            49,
            0.01,
        ),
        (
            "IMDb: flow bs2 over sizes |V|=6 (17x)",
            pick(ims, "median_sizes_over_flow_bs2", n_vars="6"),
            17,
            0.03,
        ),
        (
            "IMDb: flow bs2 over sizes |V|=10 (171x)",
            pick(ims, "median_sizes_over_flow_bs2", n_vars="10"),
            171,
            0.03,
        ),
        (
            "IMDb: 1/flow-over-entropic |V|=10 (177x)",
            1 / pick(ims, "median_flow_bs2_over_entropic", n_vars="10"),
            177,
            0.005,
        ),
        ("IMDb: checked queries (101)", pick(ims, "n_checked", n_vars="all"), 101),
        (
            "IMDb: free+l2 recovers full (116/120)",
            pick(ims, "n_free_l2_eq_full", n_vars="all"),
            116,
        ),
        (
            "IMDb: pair sizes help at most (1.35x)",
            max(float(r["entropic_over_sizes"]) for r in ime),
            1.35,
            0.005,
        ),
        (
            "IMDb: median true count |V|=6 (3e5)",
            pick(ims, "median_true_count", n_vars="6"),
            300000,
            0.03,
        ),
        (
            "IMDb: median true count |V|=10 (9e7)",
            pick(ims, "median_true_count", n_vars="10"),
            90000000,
            0.03,
        ),
        (
            "IMDb: cycles enumerated |V|=6 (364)",
            imreg["enumerated_per_length"]["cycle6"],
            364,
        ),
        (
            "IMDb: cycles enumerated |V|=8 (3,081)",
            imreg["enumerated_per_length"]["cycle8"],
            3081,
        ),
        (
            "IMDb: cycles enumerated |V|=10 (25,752)",
            imreg["enumerated_per_length"]["cycle10"],
            25752,
        ),
        (
            "IMDb superseded 2005 slice: empty joins (82/120)",
            sum(r["true_count"] == "0" for r in imsp),
            82,
        ),
        (
            "real queries below entropic at bs=2 (349)",
            pick(ims, "n_bs2_beats_entropic", n_vars="all") + 229,
            349,
        ),
        (
            "flow == polymatroid on real queries (189/240)",
            pick(ims, "n_flow_eq_poly", n_vars="all")
            + pick(fls, "n_flow_bs2_eq_poly_bs2", n_vars="all"),
            189,
        ),
        # -- review round 4: IMDb disclosure and cross-suite numbers -------
        (
            "IMDb pilot: median cover bs2/entropic (9.5e-6)",
            pick(
                load("results/imdb/superseded_sparse_2005/imdb_suite_summary.csv"),
                "median_bs2_over_entropic",
                n_vars="all",
            ),
            9.5e-6,
            0.01,
        ),
        (
            "IMDb: median cover bs2/entropic all (0.077)",
            pick(ims, "median_bs2_over_entropic", n_vars="all"),
            0.077,
            0.01,
        ),
        (
            "IMDb: 1/cover bs2 over entropic |V|=6 (4.0x)",
            1 / pick(ims, "median_bs2_over_entropic", n_vars="6"),
            4.0,
            0.01,
        ),
        (
            "IMDb: 1/cover bs2 over entropic |V|=8 (13x)",
            1 / pick(ims, "median_bs2_over_entropic", n_vars="8"),
            13,
            0.04,
        ),
        (
            "IMDb: 1/cover bs2 over entropic |V|=10 (43x)",
            1 / pick(ims, "median_bs2_over_entropic", n_vars="10"),
            43,
            0.012,
        ),
        (
            "IMDb: pair sizes help on (24) queries",
            sum(float(r["entropic_over_sizes"]) > 1.0001 for r in ime),
            24,
        ),
        (
            "IMDb: unchecked at |V|=8-10 (19 of 80)",
            sum(1 for r in ime if r["n_vars"] in ("8", "10") and r["true_count"] == ""),
            19,
        ),
        (
            "Hetionet long: max maxdeg/cover bs1 (157x)",
            max(
                float(r["improvement_maxdeg_over_bs1"])
                for r in load("results/suite/long_suite_eval.csv")
                if r.get("improvement_maxdeg_over_bs1")
            ),
            157,
            0.005,
        ),
        (
            "IMDb: max maxdeg/cover bs1 (34x)",
            max(float(r["improvement_maxdeg_over_bs1"]) for r in ime),
            34,
            0.02,
        ),
        (
            "queries with an exact count across real suites (236 = 20+115+101)",
            len(rft)
            + pick(fls, "n_checked", n_vars="all")
            + pick(ims, "n_checked", n_vars="all"),
            236,
        ),
        # -- IMDb at 5x the films (1,000, |V|=6, bounds only) --------------
        (
            "IMDb x5: bs1/entropic (4.8x)",
            pick(isc, "median_bs1_over_entropic", n_vars="all"),
            4.8,
            0.01,
        ),
        (
            "IMDb x5: 1/cover bs2 over entropic (7.0x)",
            1 / pick(isc, "median_bs2_over_entropic", n_vars="all"),
            7.0,
            0.01,
        ),
        (
            "IMDb x5: 1/flow bs2 over entropic (48x)",
            1 / pick(isc, "median_flow_bs2_over_entropic", n_vars="all"),
            48,
            0.01,
        ),
        (
            "IMDb x5: cover/flow at bs2 (6.8x)",
            pick(isc, "median_cover_bs2_over_flow_bs2", n_vars="all"),
            6.8,
            0.01,
        ),
        (
            "IMDb x5: below entropic (40/40)",
            pick(isc, "n_bs2_beats_entropic", n_vars="all"),
            40,
        ),
        (
            "IMDb x5: median stats seconds (48)",
            pick(isc, "median_stats_s", n_vars="all"),
            48,
            0.01,
        ),
        (
            "IMDb x5: max stats seconds (25 min)",
            pick(isc, "max_stats_s", n_vars="all") / 60,
            25,
            0.01,
        ),
        (
            "IMDb x5: max peak RSS (16 GB)",
            pick(isc, "max_peak_rss_mb", n_vars="all") / 1024,
            16,
            0.01,
        ),
        (
            "IMDb: median stats seconds |V|=6 (4.4)",
            pick(ims, "median_stats_s", n_vars="6"),
            4.4,
            0.02,
        ),
        (
            "IMDb: inference gap |V|=6 (3.8x)",
            pick(ims, "median_bs1_over_entropic", n_vars="6"),
            3.8,
            0.014,
        ),
        (
            "IMDb: 1/flow over entropic |V|=6 (17x)",
            1 / pick(ims, "median_flow_bs2_over_entropic", n_vars="6"),
            17,
            0.03,
        ),
        (
            "IMDb x5: data grows (4.2x rows)",
            sum(
                v["rows"]
                for v in json.loads(
                    (ROOT / "data/imdb_cycles_n1000/MANIFEST.json").read_text()
                )["relations"].values()
            )
            / sum(
                v["rows"]
                for v in json.loads(
                    (ROOT / "data/imdb_cycles/MANIFEST.json").read_text()
                )["relations"].values()
            ),
            4.2,
            0.01,
        )
        if (ROOT / "data/imdb_cycles_n1000/MANIFEST.json").exists()
        else ("IMDb x5: data grows (4.2x rows) [data absent]", 4.2, 4.2),
        # -- IMDb plan injection (DuckDB, |V|=6) ----------------------------
        (
            "IMDb injection: flow bs2 median slowdown (1.19x)",
            pick(ipi, "median_slowdown", strategy="flow_bs2"),
            1.19,
            0.005,
        ),
        (
            "IMDb injection: cover bs1 median slowdown (1.41x)",
            pick(ipi, "median_slowdown", strategy="cover_bs1"),
            1.41,
            0.005,
        ),
        (
            "IMDb injection: LpBound median slowdown (1.23x)",
            pick(ipi, "median_slowdown", strategy="lpbound"),
            1.23,
            0.005,
        ),
        (
            "IMDb injection: oracle median slowdown (1.21x)",
            pick(ipi, "median_slowdown", strategy="oracle"),
            1.21,
            0.005,
        ),
        (
            "IMDb injection: DuckDB median slowdown (0.43x)",
            pick(ipi, "median_slowdown", strategy="duckdb"),
            0.43,
            0.01,
        ),
        (
            "IMDb injection: DuckDB max slowdown (2.0x)",
            pick(ipi, "max_slowdown", strategy="duckdb"),
            2.0,
            0.01,
        ),
        (
            "IMDb injection: queries with all runs complete (33)",
            pick(ipi, "n_complete", strategy="flow_bs2"),
            33,
        ),
        (
            "IMDb injection: forced orders mostly timed out (7)",
            pick(ipi, "n_forced_mostly_timed_out", strategy="flow_bs2"),
            7,
        ),
        (
            "IMDb injection: flow bs2 picks the oracle order (24/40)",
            pick(ipi, "n_eq_oracle_order", strategy="flow_bs2"),
            24,
        ),
        (
            "IMDb injection: connected left-deep orders per 6-cycle (96)",
            max(
                int(r["n_orders"]) for r in load("results/imdb/imdb_plan_injection.csv")
            ),
            96,
        ),
        (
            "IMDb x5: median stats time grows (11x)",
            pick(isc, "median_stats_s", n_vars="all")
            / pick(ims, "median_stats_s", n_vars="6"),
            11,
            0.02,
        ),
        # -- CorrBound proxy: entropic + pair-join sizes -------------------
        (
            "CorrBound proxy: sizes over LpBound, median (31x)",
            pick(cbp, "median_bs1_over_sizes", n_vars="all"),
            31,
            0.02,
        ),
        (
            "CorrBound proxy: bs2 over sizes, median (63x)",
            pick(cbp, "median_sizes_over_full", n_vars="all"),
            63,
            0.01,
        ),
        (
            "CorrBound proxy: sizes over LpBound |V|=6 (15x)",
            pick(cbp, "median_bs1_over_sizes", n_vars="6"),
            15,
            0.02,
        ),
        (
            "CorrBound proxy: bs2 over sizes |V|=6 (19x)",
            pick(cbp, "median_sizes_over_full", n_vars="6"),
            19,
            0.025,
        ),
        (
            "CorrBound proxy: sizes over LpBound |V|=8 (27x)",
            pick(cbp, "median_bs1_over_sizes", n_vars="8"),
            27,
            0.02,
        ),
        (
            "CorrBound proxy: bs2 over sizes |V|=8 (80x)",
            pick(cbp, "median_sizes_over_full", n_vars="8"),
            80,
            0.025,
        ),
        (
            "CorrBound proxy: sizes over LpBound |V|=10 (99x)",
            pick(cbp, "median_bs1_over_sizes", n_vars="10"),
            99,
            0.02,
        ),
        (
            "CorrBound proxy: bs2 over sizes |V|=10 (154x)",
            pick(cbp, "median_sizes_over_full", n_vars="10"),
            154,
            0.025,
        ),
        (
            "CorrBound proxy: full bs2 tighter than sizes (120/120)",
            pick(cbp, "n_full_below_sizes", n_vars="all"),
            120,
        ),
        (
            "CorrBound proxy: sizes bound below true count (0)",
            pick(cbp, "n_sizes_below_true", n_vars="all"),
            0,
        ),
        # -- B4: join-free block statistics and the codegree l2 ------------
        (
            "pushdown: join-free stats mismatches on suite (0)",
            pick(pls, "n_pushdown_mismatch", n_vars="all"),
            0,
        ),
        (
            "pushdown: families + sweep mismatches (0)",
            sum(int(r["n_pushdown_mismatch"]) for r in p22 + psc),
            0,
        ),
        (
            "pushdown: s/block join build on suite (4.6)",
            pick(pls, "total_join_build_s", n_vars="all") / 91,
            4.6,
            0.01,
        ),
        (
            "pushdown: s/block join-free passes on suite (0.49)",
            pick(pls, "total_pushdown_s", n_vars="all") / 960,
            0.49,
            0.01,
        ),
        (
            "pushdown: median free/full on suite (2.3x)",
            pick(pls, "median_free_over_full", n_vars="all"),
            2.3,
            0.01,
        ),
        (
            "pushdown: max free/full on suite (18.7x)",
            pick(pls, "max_free_over_full", n_vars="all"),
            18.7,
            0.005,
        ),
        (
            "pushdown: min free/full on families (1.5x)",
            min(float(r["free_over_full"]) for r in p22),
            1.5,
            0.025,
        ),
        (
            "pushdown: max free/full on families (2.9x)",
            max(float(r["free_over_full"]) for r in p22),
            2.9,
            0.005,
        ),
        (
            "pushdown: free/full on sweep k=3 (1.9x)",
            pick(psc, "free_over_full", k="3"),
            1.9,
            0.005,
        ),
        (
            "pushdown: free/full on sweep k=16 (57x)",
            pick(psc, "free_over_full", k="16"),
            57,
            0.005,
        ),
        (
            "pushdown: free + l2 codegree == full on families (12/12)",
            sum(abs(float(r["free_l2join_over_full"]) - 1) < 1e-3 for r in p22),
            12,
        ),
        (
            "pushdown: free + l2 codegree == full on sweep (10/10)",
            sum(abs(float(r["free_l2join_over_full"]) - 1) < 1e-3 for r in psc),
            10,
        ),
        (
            "pushdown: free + l2 codegree == full on suite (85/120)",
            pick(pls, "n_free_l2join_eq_full", n_vars="all"),
            85,
        ),
        (
            "pushdown: median free+l2/full on suite (1.00)",
            pick(pls, "median_free_l2join_over_full", n_vars="all"),
            1.0,
        ),
        (
            "pushdown: max free+l2/full on suite (1.67x)",
            pick(pls, "max_free_l2join_over_full", n_vars="all"),
            1.67,
        ),
        (
            "pushdown: median free+linf/full on suite (1.70x)",
            pick(pls, "median_free_linfjoin_over_full", n_vars="all"),
            1.70,
        ),
        (
            "pushdown: median triangle-bound/full on suite (2.2x)",
            pick(pls, "median_tri_over_full", n_vars="all"),
            2.2,
            0.01,
        ),
        (
            "pushdown: median sketch/full on suite (1.04)",
            pick(pls, "median_sketch_over_full", n_vars="all"),
            1.04,
        ),
        (
            "pushdown: min sketch/full on suite (0.91)",
            pick(pls, "min_sketch_over_full", n_vars="all"),
            0.91,
        ),
        (
            "pushdown: max sketch/full on suite (1.78)",
            pick(pls, "max_sketch_over_full", n_vars="all"),
            1.78,
        ),
        (
            "pushdown: sketch below true count on suite (0)",
            pick(pls, "n_sketch_below_true", n_vars="all"),
            0,
        ),
        (
            "pushdown: s/block sketch on suite (0.32)",
            pick(pls, "total_sketch_s", n_vars="all") / 960,
            0.32,
            0.02,
        ),
        # -- C0 validity gate (SPOF) ---------------------------------------
        (
            "random validity trials (15,000)",
            sum(int(r["trials"]) for r in val) / 3,
            15000,
        ),
        ("validity violations (0)", sum(int(r["violations"]) for r in val), 0),
    ]

    for entry in checks:
        claim, actual, printed = entry[0], entry[1], entry[2]
        tol = entry[3] if len(entry) > 3 else 0.005
        ok, line = check(claim, actual, printed, tol)
        lines.append(("  OK   " if ok else "  FAIL ") + line)
        if not ok:
            failures.append(claim)

    print("Claim <-> CSV audit")
    print("=" * 78)
    print("\n".join(lines))
    print("=" * 78)
    print(
        f"  {len(checks) - len(failures)}/{len(checks)} claims match their source CSV"
    )

    if failures:
        print("\nDRIFTED CLAIMS (paper text must be updated or CSV regenerated):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("  No drift detected.")


if __name__ == "__main__":
    main()
