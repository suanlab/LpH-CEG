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
import math
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
    claim: str, actual: float, printed: float, tol: float | None = None
) -> tuple[bool, str]:
    """Claim holds if `actual` matches what the paper prints.

    With the default tolerance, counts (an integral CSV value printed as an
    integer) must match exactly and other values within 0.5%; an explicit
    `tol` is a relative tolerance for deliberately rounded magnitudes.
    """
    if tol is None and actual == round(actual) and printed == round(printed):
        ok = actual == printed
    else:
        rel = 0.005 if tol is None else tol
        ok = abs(actual - printed) <= rel * max(abs(printed), 1e-9)
    return ok, f"{claim:<52} paper={printed:<12g} csv={actual:<12.4f}"


def _tex_forms(claim: str, printed: float) -> list[str]:
    """Spellings of the claimed number as main.tex writes numbers.

    Taken from the label's parenthetical, e.g. "(614x)" or "(3,679s)",
    when it has one; otherwise from `printed`.
    """
    import re

    paren = re.findall(r"\(([^()]*)\)", claim)
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?", paren[-1]) if paren else []
    if not tokens:
        n = round(printed)
        tokens = [str(n) if printed == n else f"{printed:g}"]
    forms = []
    for t in tokens:
        plain = t.replace(",", "")
        forms += [t, t.replace(",", "{,}"), plain]
        if "." in plain:
            forms.append(plain.rstrip("0").rstrip("."))
        elif len(plain) > 3:
            forms.append(f"{int(plain):,}".replace(",", "{,}"))
            # tables print large numbers as m{\cdot}10^{e}
            e = len(plain) - 1
            forms.append(f"{int(plain) / 10**e:g}{{\\cdot}}10^{{{e}}}")
    return forms


def missing_from_paper(checks: list[tuple]) -> list[str]:  # type: ignore[type-arg]
    """Pinned values whose printed number does not occur in main.tex.

    Non-fatal: a number can legitimately appear only in a figure or in a
    derived form; the list exists to catch pins that outlived their text.
    Pins labelled "internal: ..." guard a CSV fact the paper relies on
    without printing it, and are not expected in main.tex.
    """
    import re

    paper = ROOT / "docs" / "paper"
    # numbers moved to the extended version still count as printed
    tex = (paper / "main.tex").read_text() + (
        paper / "extended_version.tex"
    ).read_text()
    out = []
    for entry in checks:
        claim, printed = entry[0], float(entry[2])
        if claim.startswith("internal: "):
            continue
        forms = _tex_forms(claim, printed)
        if not any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", tex) for f in forms):
            out.append(f"{claim} [{forms[-1]}]")
    return out


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
    ordr = load("results/wcoj/ordering_study_summary.csv")
    pv = load("results/ablation/pvalue_ablation.csv")
    sub = load("results/ablation/substitutes_ablation.csv")
    val = load("results/validity/validity_search.csv")
    suite = load("results/suite/suite_summary.csv")
    cost = load("results/cost/stats_cost.csv")
    shannon = load("results/validity/shannon_implication.csv")
    wide = load("results/validity/shannon_wide.csv")
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
    pfs = load("results/suite/pset_fair_summary.csv")
    pfr = load("results/suite/pset_fair.csv")
    p11s = load("results/suite/p11_short.csv")
    p11f = load("results/flow/p11_2x2.csv")
    gcs = load("results/gcare/gcare_summary.csv")
    gce = load("results/gcare/gcare_eval.csv")
    gcr = load("results/gcare/gcare_recount.csv")
    slr = load("results/suite/slice_robustness_summary.csv")
    lse = load("results/suite/long_suite_eval.csv")
    sco = load("results/cost/stats_cost.csv")
    lpv = load("results/scalability/lp_pvalues.csv")
    pcs = load("results/suite/poly_certify_summary.csv")
    gpc = load("results/gcare/gcare_poly_certify_summary.csv")
    spc = load("results/suite/slice_poly_certify_summary.csv")
    bci = load("results/suite/bootstrap_ci.csv")
    phl = load("results/suite/poly_headline.csv")
    pce = load("results/suite/poly_certify.csv")
    gpe = load("results/gcare/gcare_poly_certify.csv")
    spe = load("results/suite/slice_poly_certify.csv")
    pse = load("results/suite/poly_certify_short.csv")
    swc = load("results/flow/sweep_poly_certify.csv")
    sps = load("results/spectral/spectral_summary.csv")
    osg = load("results/characterization/opt_stats_gap_summary.csv")
    vsl = load("results/characterization/voi_selection_summary.csv")

    def ph(suite: str, group: str, col: str) -> float:
        return pick(phl, col, suite=suite, group=group)

    jfu = load("results/large_scale/job_full.csv")
    pbs = load("results/imdb/predicate_blocks_summary.csv")
    rrs = load("results/e2e/robustness_repeat_summary.csv")
    apx = load("results/pushdown/ablation_p11_poly_summary.csv")
    cac = load("results/cost/cost_accuracy_summary.csv")
    cai = load("results/cost/cost_accuracy_indep_summary.csv")
    abo = load("results/pushdown/ablation_overlap_summary.csv")
    rrw = load("results/e2e/robustness_raw_summary.csv")
    iss = load("results/imdb/imdb_scale_sparse_summary.csv")
    alp = load("results/corrbound/sm_artifact_lp_summary.csv")
    cbc = load("results/corrbound/sm_combined_summary.csv")
    swc = load("results/flow/sweep_poly_certify.csv")
    cbs = load("results/corrbound/sm_compare_summary.csv")
    vcs = load("results/characterization/voi_cost_summary.csv")
    ssc = load("results/suite/slice_scale_summary.csv")
    ssr = load("results/suite/slice_scale.csv")
    vcr = load("results/characterization/voi_cost.csv")
    pbr = load("results/imdb/predicate_blocks.csv")
    prk = load("results/cost/plan_ranking.csv")

    def gc(ds: str, col: str) -> float:
        return pick(gcs, col, dataset=ds)

    def med_ratio(rows: list[dict[str, str]], a: str, b: str, **where: str) -> float:
        rs = [r for r in rows if all(r[k] == v for k, v in where.items())]
        return statistics.median(float(r[a]) / float(r[b]) for r in rs)

    def rng_ratio(rows: list[dict[str, str]], a: str, b: str) -> tuple[float, float]:
        v = [float(r[a]) / float(r[b]) for r in rows]
        return min(v), max(v)

    long_true = {
        r["query_id"]: r["true_count"]
        for r in load("results/suite/long_suite_eval.csv")
    }
    long_true.update(
        {
            r["query_id"]: r["true_count"]
            for r in load("results/imdb/imdb_suite_eval.csv")
        }
    )

    def p11_over_true(suite: str, nv: str, col: str) -> float:
        rs = [
            r
            for r in pfr
            if r["suite"] == suite
            and r["n_vars"] == nv
            and long_true[r["query_id"]] not in ("", "0")
        ]
        return statistics.median(
            float(r[col]) / int(long_true[r["query_id"]]) for r in rs
        )

    def pf(suite: str, nv: str, col: str) -> float:
        return pick(pfs, col, suite=suite, n_vars=nv)

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
        (
            "entropic LP at k=12 (25 s, sparse, interior-point)",
            pick(se, "entropic_ms", k="12") / 1000.0,
            25,
            0.5 / 25,
        ),
        (
            "entropic LP at k=14 (2,480 s -- NOT intractable)",
            pick(se, "entropic_ms", k="14") / 1000.0,
            2480,
            0.5 / 2480,
        ),
        (
            "max synthetic cover/entropic gap (1.43x)",
            max(float(r["gap_cover_over_entropic"]) for r in se if r["status"] == "ok"),
            1.4293,
        ),
        (
            "real crossover k=12 entropic (22 s, sparse, interior-point)",
            pick(cr, "entropic_ms", k="12") / 1000.0,
            22,
            0.5 / 22,
        ),
        (
            "real crossover k=14 entropic (27 min -- solves)",
            pick(cr, "entropic_ms", k="14") / 60000.0,
            27,
            0.5 / 27,
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
            "p beyond {1,2,inf} adds (<2.4%), rounded up to 0.1%",
            math.ceil(
                max(
                    (float(r["{1,1.5,2,3,5,inf}"]) / float(r["{1,2,inf}"]) - 1) * 100
                    for r in pv
                )
                * 10
            )
            / 10,
            2.4,
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
            "internal: Shannon-cone checks all certified",
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
            "cover/block beats entropic/single on all pairs (12/12, flow_2x2)",
            sum(float(r["cover_bs2"]) < float(r["entropic_bs1"]) for r in f22),
            12,
        ),
        (
            "optimum wins at EQUAL statistics on all pairs (12/12, flow_2x2)",
            sum(float(r["poly_bs2"]) < float(r["cover_bs2"]) for r in f22),
            12,
        ),
        (
            "refinement: cutting planes close 0% of the gap on 5/5 instances",
            sum(
                float(r["gap_closed_frac"]) == 0.0
                for r in load("results/cost/refinement_hierarchy.csv")
            ),
            5,
        ),
        (
            "refinement: cycle-4 cover bound log2 (15.335)",
            math.log2(
                pick(
                    load("results/cost/refinement_hierarchy.csv"),
                    "cover_bound",
                    family="cycle4",
                )
            ),
            15.335,
            0.0001,
        ),
        (
            "refinement: cycle-4 entropic bound log2 (14.858)",
            math.log2(
                pick(
                    load("results/cost/refinement_hierarchy.csv"),
                    "entropic_bound",
                    family="cycle4",
                )
            ),
            14.858,
            0.0001,
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
            "AGM over pair-join sizes below entropic, smallest median (7x)",
            min(1 / float(r["median_agm_pairjoins_over_entropic"]) for r in apj),
            7,
            0.5 / 7,
        ),
        (
            "AGM over pair-join sizes below entropic, largest median (15x)",
            max(1 / float(r["median_agm_pairjoins_over_entropic"]) for r in apj),
            15,
            0.5 / 15,
        ),
        (
            "internal: bs2/AGM+pairjoins median |V|=6 (0.22)",
            pick(apj, "median_bs2_over_agm_pairjoins", n_vars="6"),
            0.22,
            0.01,
        ),
        (
            "internal: bs2/AGM+pairjoins median |V|=10 (0.0047)",
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
            "entropic 2^|V| LP at |V|=8 (81 ms)",
            pick(sent, "entropic_ms", k="8"),
            81,
            0.5 / 81,
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
            "real frontier: all 109 checked against exact counts, all valid",
            sum(r["valid"] == "True" for r in rft),
            109,
        ),
        (
            "real frontier: max true count (334)",
            max(int(r["true_count"]) for r in rft),
            334,
        ),
        (
            "real frontier: median true count (30)",
            statistics.median(int(r["true_count"]) for r in rft),
            30,
        ),
        (
            "real frontier: median bs2/true over positive counts (85x)",
            statistics.median(
                float(r["bs2"]) / int(r["true_count"])
                for r in rft
                if int(r["true_count"]) > 0
            ),
            85,
            0.01,
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
            "internal: IMDb: median_flow_bs2_over_entropic |V|=8 (0.019)",
            pick(ims, "median_flow_bs2_over_entropic", n_vars="8"),
            0.019,
            0.03,
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
            "queries with an exact count across real suites (325 = 109+115+101)",
            len(rft)
            + pick(fls, "n_checked", n_vars="all")
            + pick(ims, "n_checked", n_vars="all"),
            325,
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
        # -- eleven norms on both sides (pset_fair) ---------------------------
        (
            "P11 both sides: hetionet gain |V|=6 (1.9)",
            pf("hetionet", "6", "median_entropic_p3_over_p11"),
            1.9,
            0.03,
        ),
        (
            "P11 both sides: hetionet gain |V|=8 (2.7)",
            pf("hetionet", "8", "median_entropic_p3_over_p11"),
            2.7,
            0.03,
        ),
        (
            "P11 both sides: hetionet gain |V|=10 (3.2)",
            pf("hetionet", "10", "median_entropic_p3_over_p11"),
            3.2,
            0.03,
        ),
        (
            "P11 both sides: hetionet c1 |V|=6 (2.66)",
            pf("hetionet", "6", "median_cover_bs1_over_entropic_p11"),
            2.66,
            0.005,
        ),
        (
            "P11 both sides: hetionet c1 |V|=8 (3.1)",
            pf("hetionet", "8", "median_cover_bs1_over_entropic_p11"),
            3.1,
            0.005,
        ),
        (
            "P11 both sides: hetionet c1 |V|=10 (3.0)",
            pf("hetionet", "10", "median_cover_bs1_over_entropic_p11"),
            3.0,
            0.005,
        ),
        (
            "P11 both sides: hetionet c2 |V|=6 (0.022)",
            pf("hetionet", "6", "median_cover_bs2_over_entropic_p11"),
            0.022,
            0.05,
        ),
        (
            "P11 both sides: hetionet c2 |V|=8 (0.0037)",
            pf("hetionet", "8", "median_cover_bs2_over_entropic_p11"),
            0.0037,
            0.05,
        ),
        (
            "P11 both sides: hetionet c2 |V|=10 (0.0011)",
            pf("hetionet", "10", "median_cover_bs2_over_entropic_p11"),
            0.0011,
            0.05,
        ),
        (
            "P11 both sides: hetionet f2 |V|=6 (0.0042)",
            pf("hetionet", "6", "median_flow_bs2_over_entropic_p11"),
            0.0042,
            0.05,
        ),
        (
            "P11 both sides: hetionet below |V|=6 (40)",
            pf("hetionet", "6", "n_cover_bs2_below_entropic_p11"),
            40,
        ),
        (
            "P11 both sides: hetionet below |V|=8 (40)",
            pf("hetionet", "8", "n_cover_bs2_below_entropic_p11"),
            40,
        ),
        (
            "P11 both sides: hetionet below |V|=10 (40)",
            pf("hetionet", "10", "n_cover_bs2_below_entropic_p11"),
            40,
        ),
        (
            "P11 both sides: imdb gain |V|=6 (3.5)",
            pf("imdb", "6", "median_entropic_p3_over_p11"),
            3.5,
            0.03,
        ),
        (
            "P11 both sides: imdb gain |V|=8 (5.1)",
            pf("imdb", "8", "median_entropic_p3_over_p11"),
            5.1,
            0.03,
        ),
        (
            "P11 both sides: imdb gain |V|=10 (6.7)",
            pf("imdb", "10", "median_entropic_p3_over_p11"),
            6.7,
            0.03,
        ),
        (
            "P11 both sides: imdb c1 |V|=6 (5.14)",
            pf("imdb", "6", "median_cover_bs1_over_entropic_p11"),
            5.14,
            0.005,
        ),
        (
            "P11 both sides: imdb c1 |V|=8 (6.75)",
            pf("imdb", "8", "median_cover_bs1_over_entropic_p11"),
            6.75,
            0.005,
        ),
        (
            "P11 both sides: imdb c1 |V|=10 (5.96)",
            pf("imdb", "10", "median_cover_bs1_over_entropic_p11"),
            5.96,
            0.005,
        ),
        (
            "P11 both sides: imdb c2 |V|=6 (0.79)",
            pf("imdb", "6", "median_cover_bs2_over_entropic_p11"),
            0.79,
            0.05,
        ),
        (
            "P11 both sides: imdb c2 |V|=8 (0.35)",
            pf("imdb", "8", "median_cover_bs2_over_entropic_p11"),
            0.35,
            0.05,
        ),
        (
            "P11 both sides: imdb c2 |V|=10 (0.15)",
            pf("imdb", "10", "median_cover_bs2_over_entropic_p11"),
            0.15,
            0.05,
        ),
        (
            "internal: P11 both sides: imdb f2 |V|=6 (0.11)",
            pf("imdb", "6", "median_flow_bs2_over_entropic_p11"),
            0.11,
            0.05,
        ),
        (
            "internal: P11 both sides: imdb f2 |V|=8 (0.034)",
            pf("imdb", "8", "median_flow_bs2_over_entropic_p11"),
            0.034,
            0.05,
        ),
        (
            "internal: P11 both sides: imdb f2 |V|=10 (0.012)",
            pf("imdb", "10", "median_flow_bs2_over_entropic_p11"),
            0.012,
            0.05,
        ),
        (
            "P11 both sides: imdb below |V|=6 (32)",
            pf("imdb", "6", "n_cover_bs2_below_entropic_p11"),
            32,
        ),
        (
            "P11 both sides: imdb below |V|=8 (40)",
            pf("imdb", "8", "n_cover_bs2_below_entropic_p11"),
            40,
        ),
        (
            "P11 both sides: imdb below |V|=10 (40)",
            pf("imdb", "10", "n_cover_bs2_below_entropic_p11"),
            40,
        ),
        (
            "P11: Hetionet cover bs2 gain |V|=6 (46x)",
            1 / pf("hetionet", "6", "median_cover_bs2_over_entropic_p11"),
            46,
            0.01,
        ),
        (
            "P11: Hetionet cover bs2 gain |V|=8 (270x)",
            1 / pf("hetionet", "8", "median_cover_bs2_over_entropic_p11"),
            270,
            0.01,
        ),
        (
            "P11: Hetionet cover bs2 gain |V|=10 (920x)",
            1 / pf("hetionet", "10", "median_cover_bs2_over_entropic_p11"),
            920,
            0.005,
        ),
        (
            "P11: IMDb cover bs2 gain |V|=6 (1.3x)",
            1 / pf("imdb", "6", "median_cover_bs2_over_entropic_p11"),
            1.3,
            0.04,
        ),
        (
            "P11: IMDb cover bs2 gain |V|=8 (2.9x)",
            1 / pf("imdb", "8", "median_cover_bs2_over_entropic_p11"),
            2.9,
            0.01,
        ),
        (
            "P11: IMDb cover bs2 gain |V|=10 (6.8x)",
            1 / pf("imdb", "10", "median_cover_bs2_over_entropic_p11"),
            6.8,
            0.01,
        ),
        (
            "P11: IMDb flow bs2 below entropic (120/120)",
            pf("imdb", "all", "n_flow_bs2_below_entropic_p11"),
            120,
        ),
        (
            "P11: P3 rows reproduce the paper (entropic, max rel dev 0)",
            max(
                abs(float(r["entropic_p3"]) / float(o["entropic"]) - 1)
                for r in pfr
                if r["suite"] == "imdb"
                for o in ime
                if o["query_id"] == r["query_id"]
            ),
            0,
        ),
        (
            "P11: bounds with an exact count, all valid (864 = 216 x 4)",
            sum(
                float(r[c]) >= int(t)
                for r in pfr
                for t in [
                    next(
                        (
                            o["true_count"]
                            for o in (
                                load("results/suite/long_suite_eval.csv")
                                if r["suite"] == "hetionet"
                                else ime
                            )
                            if o["query_id"] == r["query_id"]
                        ),
                        "",
                    )
                ]
                if t != ""
                for c in (
                    "entropic_p11",
                    "cover_bs1_p11",
                    "cover_bs2_p11",
                    "flow_bs2_p11",
                )
            ),
            864,
        ),
        # -- eleven norms as the primary comparison (p11 reruns) --------------
        (
            "short P11: stats gain cover bs1/bs2 (30x)",
            med_ratio(p11s, "cover_bs1_p11", "cover_bs2_p11"),
            30,
            0.005,
        ),
        (
            "short P3: stats gain cover bs1/bs2 (56x)",
            med_ratio(p11s, "cover_bs1_p3", "cover_bs2_p3"),
            56,
            0.005,
        ),
        (
            "short P11: inference gap cover bs1/entropic (2.8x)",
            med_ratio(p11s, "cover_bs1_p11", "entropic_p11"),
            2.8,
            0.015,
        ),
        (
            "short P3: inference gap (2.6x)",
            med_ratio(p11s, "cover_bs1_p3", "entropic_p3"),
            2.6,
            0.01,
        ),
        (
            "short P11: cover bs2/entropic (0.094)",
            med_ratio(p11s, "cover_bs2_p11", "entropic_p11"),
            0.094,
            0.005,
        ),
        (
            "short P3: cover bs2/entropic (0.057)",
            med_ratio(p11s, "cover_bs2_p3", "entropic_p3"),
            0.057,
            0.01,
        ),
        (
            "short P11: flow bs2 below entropic (109/109)",
            sum(float(r["flow_bs2_p11"]) < float(r["entropic_p11"]) for r in p11s),
            109,
        ),
        (
            "349 = 109 + 240 below entropic under the flow LP (P11)",
            sum(float(r["flow_bs2_p11"]) < float(r["entropic_p11"]) for r in p11s)
            + sum(float(r["flow_bs2_p11"]) < float(r["entropic_p11"]) for r in pfr),
            349,
        ),
        (
            "2x2 P11 table triangle cover_bs1_p11 (1083)",
            pick(p11f, "cover_bs1_p11", family="triangle", seed="101"),
            1083,
            0.002,
        ),
        (
            "2x2 P11 table triangle cover_bs2_p11 (668)",
            pick(p11f, "cover_bs2_p11", family="triangle", seed="101"),
            668,
            0.002,
        ),
        (
            "2x2 P11 table triangle entropic_p11 (806)",
            pick(p11f, "entropic_p11", family="triangle", seed="101"),
            806,
            0.002,
        ),
        (
            "2x2 P11 table triangle poly_bs2_p11 (352)",
            pick(p11f, "poly_bs2_p11", family="triangle", seed="101"),
            352,
            0.002,
        ),
        (
            "2x2 P11 table cycle4 cover_bs1_p11 (9855)",
            pick(p11f, "cover_bs1_p11", family="cycle4", seed="101"),
            9855,
            0.002,
        ),
        (
            "2x2 P11 table cycle4 cover_bs2_p11 (4716)",
            pick(p11f, "cover_bs2_p11", family="cycle4", seed="101"),
            4716,
            0.002,
        ),
        (
            "2x2 P11 table cycle4 entropic_p11 (7388)",
            pick(p11f, "entropic_p11", family="cycle4", seed="101"),
            7388,
            0.002,
        ),
        (
            "2x2 P11 table cycle4 poly_bs2_p11 (1355)",
            pick(p11f, "poly_bs2_p11", family="cycle4", seed="101"),
            1355,
            0.002,
        ),
        (
            "2x2 P11 table diamond cover_bs1_p11 (9769)",
            pick(p11f, "cover_bs1_p11", family="diamond", seed="101"),
            9769,
            0.002,
        ),
        (
            "2x2 P11 table diamond cover_bs2_p11 (4281)",
            pick(p11f, "cover_bs2_p11", family="diamond", seed="101"),
            4281,
            0.002,
        ),
        (
            "2x2 P11 table diamond entropic_p11 (7388)",
            pick(p11f, "entropic_p11", family="diamond", seed="101"),
            7388,
            0.002,
        ),
        (
            "2x2 P11 table diamond poly_bs2_p11 (1252)",
            pick(p11f, "poly_bs2_p11", family="diamond", seed="101"),
            1252,
            0.002,
        ),
        (
            "2x2 P11 table k4 cover_bs1_p11 (9125)",
            pick(p11f, "cover_bs1_p11", family="k4", seed="101"),
            9125,
            0.002,
        ),
        (
            "2x2 P11 table k4 cover_bs2_p11 (3376)",
            pick(p11f, "cover_bs2_p11", family="k4", seed="101"),
            3376,
            0.002,
        ),
        (
            "2x2 P11 table k4 entropic_p11 (6320)",
            pick(p11f, "entropic_p11", family="k4", seed="101"),
            6320,
            0.002,
        ),
        (
            "2x2 P11 table k4 poly_bs2_p11 (1093)",
            pick(p11f, "poly_bs2_p11", family="k4", seed="101"),
            1093,
            0.002,
        ),
        (
            "2x2 P11 stats cover min (1.5)",
            rng_ratio(p11f, "cover_bs1_p11", "cover_bs2_p11")[0],
            1.5,
            0.034,
        ),
        (
            "2x2 P11 stats cover max (2.7)",
            rng_ratio(p11f, "cover_bs1_p11", "cover_bs2_p11")[1],
            2.7,
            0.01,
        ),
        (
            "2x2 P11 stats opt min (1.8)",
            rng_ratio(p11f, "entropic_p11", "poly_bs2_p11")[0],
            1.8,
            0.01,
        ),
        (
            "2x2 P11 stats opt max (5.9)",
            rng_ratio(p11f, "entropic_p11", "poly_bs2_p11")[1],
            5.9,
            0.01,
        ),
        (
            "2x2 P11 inference single min (1.2)",
            rng_ratio(p11f, "cover_bs1_p11", "entropic_p11")[0],
            1.2,
            0.01,
        ),
        (
            "2x2 P11 inference single max (1.5)",
            rng_ratio(p11f, "cover_bs1_p11", "entropic_p11")[1],
            1.5,
            0.01,
        ),
        (
            "2x2 P11 inference block min (1.6)",
            rng_ratio(p11f, "cover_bs2_p11", "poly_bs2_p11")[0],
            1.6,
            0.03,
        ),
        (
            "2x2 P11 inference block max (3.8)",
            rng_ratio(p11f, "cover_bs2_p11", "poly_bs2_p11")[1],
            3.8,
            0.014,
        ),
        (
            "2x2 P11 flow == polymatroid (7/12)",
            sum(
                abs(float(r["flow_bs2_p11"]) / float(r["poly_bs2_p11"]) - 1) < 1e-3
                for r in p11f
            ),
            7,
        ),
        (
            "sweep: cover/entropic identical with P11 (max dev 0)",
            max(
                abs(
                    float(r["cover_bs1_p11"]) / float(r["entropic_p11"])
                    - float(r["cover_bs1_p3"]) / float(r["entropic_p3"])
                )
                for r in load("results/scalability/p11_sweep.csv")
            ),
            0,
        ),
        (
            "P11 entropic/exact hetionet |V|=6 (8400)",
            p11_over_true("hetionet", "6", "entropic_p11"),
            8400,
            0.025,
        ),
        (
            "P11 flow bs2/exact hetionet |V|=6 (47)",
            p11_over_true("hetionet", "6", "flow_bs2_p11"),
            47,
            0.025,
        ),
        (
            "P11 entropic/exact hetionet |V|=8 (360000)",
            p11_over_true("hetionet", "8", "entropic_p11"),
            360000,
            0.025,
        ),
        (
            "P11 flow bs2/exact hetionet |V|=8 (277)",
            p11_over_true("hetionet", "8", "flow_bs2_p11"),
            277,
            0.025,
        ),
        (
            "P11 entropic/exact hetionet |V|=10 (28000000)",
            p11_over_true("hetionet", "10", "entropic_p11"),
            28000000,
            0.025,
        ),
        (
            "P11 entropic/exact imdb |V|=6 (23)",
            p11_over_true("imdb", "6", "entropic_p11"),
            23,
            0.025,
        ),
        (
            "P11 entropic/exact imdb |V|=8 (171)",
            p11_over_true("imdb", "8", "entropic_p11"),
            171,
            0.025,
        ),
        (
            "P11 entropic/exact imdb |V|=10 (1300)",
            p11_over_true("imdb", "10", "entropic_p11"),
            1300,
            0.025,
        ),
        (
            "P11 families cover/flow bs2 min (1.6x)",
            rng_ratio(p11f, "cover_bs2_p11", "flow_bs2_p11")[0],
            1.6,
            0.031,
        ),
        (
            "P11 families cover/flow bs2 max (3.8x)",
            rng_ratio(p11f, "cover_bs2_p11", "flow_bs2_p11")[1],
            3.8,
            0.014,
        ),
        # -- G-CARE cyclic queries (external benchmark) ---------------------
        ("G-CARE aids n_cyclic (241)", gc("aids", "n_cyclic"), 241),
        ("G-CARE aids n_evaluated (241)", gc("aids", "n_evaluated"), 241),
        (
            "G-CARE aids n_gcare_true_eq_exact (241)",
            gc("aids", "n_gcare_true_eq_exact"),
            241,
        ),
        (
            "G-CARE aids n_cover_bs2_below_entropic_p11 (135)",
            gc("aids", "n_cover_bs2_below_entropic_p11"),
            135,
        ),
        (
            "G-CARE aids n_flow_bs2_below_entropic_p11 (241)",
            gc("aids", "n_flow_bs2_below_entropic_p11"),
            241,
        ),
        ("G-CARE aids n_invalid_p11 (0)", gc("aids", "n_invalid_p11"), 0),
        ("G-CARE yago n_cyclic (366)", gc("yago", "n_cyclic"), 366),
        ("G-CARE yago n_evaluated (192)", gc("yago", "n_evaluated"), 192),
        ("G-CARE yago n_too_large (144)", gc("yago", "n_too_large"), 144),
        (
            "G-CARE yago n_pair_join_too_large (30)",
            gc("yago", "n_pair_join_too_large"),
            30,
        ),
        (
            "G-CARE yago n_cover_bs2_below_entropic_p11 (177)",
            gc("yago", "n_cover_bs2_below_entropic_p11"),
            177,
        ),
        (
            "G-CARE yago n_flow_bs2_below_entropic_p11 (188)",
            gc("yago", "n_flow_bs2_below_entropic_p11"),
            188,
        ),
        ("G-CARE yago n_invalid_p11 (0)", gc("yago", "n_invalid_p11"), 0),
        (
            "G-CARE aids median_cover_bs1_over_entropic_p11 (59)",
            gc("aids", "median_cover_bs1_over_entropic_p11"),
            59,
            0.01,
        ),
        (
            "G-CARE aids median_cover_bs2_over_entropic_p11 (0.74)",
            gc("aids", "median_cover_bs2_over_entropic_p11"),
            0.74,
            0.02,
        ),
        (
            "G-CARE aids median_entropic_over_exact_p11 (513)",
            gc("aids", "median_entropic_over_exact_p11"),
            513,
            0.001,
        ),
        (
            "G-CARE aids median_flow_bs2_over_exact_p11 (66)",
            gc("aids", "median_flow_bs2_over_exact_p11"),
            66,
            0.005,
        ),
        (
            "G-CARE yago median_cover_bs1_over_entropic_p11 (6.1)",
            gc("yago", "median_cover_bs1_over_entropic_p11"),
            6.1,
            0.01,
        ),
        (
            "G-CARE yago median_cover_bs2_over_entropic_p11 (0.11)",
            gc("yago", "median_cover_bs2_over_entropic_p11"),
            0.11,
            0.046,
        ),
        (
            "G-CARE yago median_entropic_over_exact_p11 (2139)",
            gc("yago", "median_entropic_over_exact_p11"),
            2139,
            0.001,
        ),
        (
            "slice prefix: cover bs2 below LpBound at |V|=6 (46x)",
            1 / pick(slr, "median_cover_bs2_over_entropic_p11", slice="prefix"),
            46,
            0.5 / 46,
        ),
        (
            "slice prefix: flow bs2 below LpBound at |V|=6 (236x)",
            1 / pick(slr, "median_flow_bs2_over_entropic_p11", slice="prefix"),
            236,
            0.5 / 236,
        ),
        (
            "slice prefix: flow bs2 below LpBound on (40/40)",
            pick(slr, "n_flow_bs2_below_entropic_p11", slice="prefix"),
            40,
        ),
        (
            "slice 1: cover bs2 below LpBound at |V|=6 (46x)",
            1 / pick(slr, "median_cover_bs2_over_entropic_p11", slice="1"),
            46,
            0.5 / 46,
        ),
        (
            "slice 1: flow bs2 below LpBound at |V|=6 (233x)",
            1 / pick(slr, "median_flow_bs2_over_entropic_p11", slice="1"),
            233,
            0.5 / 233,
        ),
        (
            "slice 1: flow bs2 below LpBound on (40/40)",
            pick(slr, "n_flow_bs2_below_entropic_p11", slice="1"),
            40,
        ),
        (
            "slice 2: cover bs2 below LpBound at |V|=6 (49x)",
            1 / pick(slr, "median_cover_bs2_over_entropic_p11", slice="2"),
            49,
            0.5 / 49,
        ),
        (
            "slice 2: flow bs2 below LpBound at |V|=6 (244x)",
            1 / pick(slr, "median_flow_bs2_over_entropic_p11", slice="2"),
            244,
            0.5 / 244,
        ),
        (
            "slice 2: flow bs2 below LpBound on (40/40)",
            pick(slr, "n_flow_bs2_below_entropic_p11", slice="2"),
            40,
        ),
        (
            "Fig.1 caption: cover LP end to end, min (7 ms)",
            min(float(r["cover_ms"]) for r in se),
            7,
            0.5 / 7,
        ),
        (
            "Fig.1 caption: cover LP end to end, max (43 ms)",
            max(float(r["cover_ms"]) for r in se),
            43,
            0.5 / 43,
        ),
        (
            "certified: short queries checked (109)",
            pick(pcs, "n", suite="short"),
            109,
            0,
        ),
        (
            "certified: short flow below polymatroid (0)",
            pick(pcs, "n_flow_below_poly", suite="short"),
            0,
            0,
        ),
        (
            "certified: hetionet queries checked (120)",
            pick(pcs, "n", suite="hetionet"),
            120,
            0,
        ),
        (
            "certified: hetionet flow below polymatroid (0)",
            pick(pcs, "n_flow_below_poly", suite="hetionet"),
            0,
            0,
        ),
        (
            "certified: imdb queries checked (120)",
            pick(pcs, "n", suite="imdb"),
            120,
            0,
        ),
        (
            "certified: imdb flow below polymatroid (0)",
            pick(pcs, "n_flow_below_poly", suite="imdb"),
            0,
            0,
        ),
        (
            "plan ranking: cover tau below entropic on (4) families",
            sum(
                float(r["tau_cover_vs_true"]) < float(r["tau_entropic_vs_true"])
                for r in prk
            ),
            4,
        ),
        (
            "plan ranking: cover tau min (0.27)",
            min(float(r["tau_cover_vs_true"]) for r in prk),
            0.27,
            0.005 / 0.27,
        ),
        (
            "plan ranking: cover tau max (0.87)",
            max(float(r["tau_cover_vs_true"]) for r in prk),
            0.87,
            0.005 / 0.87,
        ),
        (
            "plan ranking: entropic tau min (0.33)",
            min(float(r["tau_entropic_vs_true"]) for r in prk),
            0.33,
            0.005 / 0.33,
        ),
        (
            "plan ranking: entropic tau max (0.94)",
            max(float(r["tau_entropic_vs_true"]) for r in prk),
            0.94,
            0.005 / 0.94,
        ),
        (
            "30K-row prefix: cover bs2 below LpBound at |V|=6 (46x)",
            pick(ssc, "median_entropic_over_cover_bs2_p11", cap="30000"),
            46,
            0.5 / 46,
        ),
        (
            "30K-row prefix: cover bs2 below LpBound on (40)/40",
            pick(ssc, "n_cover_bs2_below_entropic_p11", cap="30000"),
            40,
        ),
        (
            "30K-row prefix: queries with a relation over 20K rows (39)",
            sum(
                int(r["max_relation_rows"]) > 20000 for r in ssr if r["cap"] == "30000"
            ),
            39,
        ),
        (
            "internal: 30K-row prefix: invalid bounds",
            sum(int(r["n_invalid"]) for r in ssc),
            0,
        ),
        (
            "internal: 30K-row prefix: 20K reference reproduces Table long-suite",
            pick(ssc, "median_entropic_over_cover_bs2_p11", cap="20000"),
            1 / ph("hetionet", "6", "median_cover2_over_entropic"),
            0.005,
        ),
        (
            "tail repeat: bound-guided P95 (1.18x)",
            pick(rrs, "bound_p95", scope="median-of-5 timings"),
            1.18,
            0.005 / 1.18,
        ),
        (
            "tail repeat: bound-guided max (1.21x)",
            pick(rrs, "bound_max", scope="median-of-5 timings"),
            1.21,
            0.005 / 1.21,
        ),
        (
            "tail repeat: DuckDB P95 (5.51x)",
            pick(rrs, "duckdb_p95", scope="median-of-5 timings"),
            5.51,
            0.005 / 5.51,
        ),
        (
            "tail repeat: DuckDB max (6.77x)",
            pick(rrs, "duckdb_max", scope="median-of-5 timings"),
            6.77,
            0.005 / 6.77,
        ),
        (
            "tail repeat: medians bound/DuckDB (1.03x)",
            pick(rrs, "bound_median", scope="median-of-5 timings"),
            1.03,
            0.005 / 1.03,
        ),
        (
            "tail repeat: DuckDB median (1.18x) [same value as bound P95]",
            pick(rrs, "duckdb_median", scope="median-of-5 timings"),
            1.18,
            0.005 / 1.18,
        ),
        (
            "tail repeat: worst per-rep bound-guided P95 (1.65x)",
            max(float(r["bound_p95"]) for r in rrs if r["scope"].startswith("rep")),
            1.65,
            0.005 / 1.65,
        ),
        (
            "tail repeat: best per-rep DuckDB P95 (5.07x)",
            min(float(r["duckdb_p95"]) for r in rrs if r["scope"].startswith("rep")),
            5.07,
            0.005 / 5.07,
        ),
        (
            "headline ablation: join-free share of log-gain, hetionet (88%)",
            100 * pick(apx, "median_free_gain_share", suite="hetionet", n_vars="all"),
            88,
            0.5 / 88,
        ),
        (
            "headline ablation: join-free share of log-gain, imdb (86%)",
            100 * pick(apx, "median_free_gain_share", suite="imdb", n_vars="all"),
            86,
            0.5 / 86,
        ),
        (
            "headline ablation: free+l2 over full, hetionet (1.47x)",
            pick(apx, "median_free_l2_over_full", suite="hetionet", n_vars="all"),
            1.47,
            0.005 / 1.47,
        ),
        (
            "headline ablation: free+l2 over full, imdb (1.39x)",
            pick(apx, "median_free_l2_over_full", suite="imdb", n_vars="all"),
            1.39,
            0.005 / 1.39,
        ),
        (
            "headline ablation: free+l2 recovers full, hetionet (1)/120",
            pick(apx, "n_free_l2_eq_full", suite="hetionet", n_vars="all"),
            1,
        ),
        (
            "headline ablation: free+l2 recovers full, imdb (1)/120",
            pick(apx, "n_free_l2_eq_full", suite="imdb", n_vars="all"),
            1,
        ),
        (
            "CorrBound workload: queries (400)",
            pick(cbs, "n", method="pairwise"),
            400,
        ),
        (
            "CorrBound workload: our LpBound / artifact, min (1.0)",
            pick(cbs, "lpbound_ours_over_artifact_min", method="pairwise"),
            1.0,
            1e-06 / 1.0,
        ),
        (
            "CorrBound workload: our LpBound / artifact, max (1.0)",
            pick(cbs, "lpbound_ours_over_artifact_max", method="pairwise"),
            1.0,
            1e-06 / 1.0,
        ),
        (
            "CorrBound workload: block median over exact (1.99)",
            pick(cbs, "median_over_truth", method="block_poly"),
            1.99,
            0.005 / 1.99,
        ),
        (
            "CorrBound workload: CorrBound-e median over exact (6.1)",
            pick(cbs, "median_over_truth", method="corrbound_e"),
            6.1,
            0.05 / 6.1,
        ),
        (
            "CorrBound workload: LpBound median over exact (64.6)",
            pick(cbs, "median_over_truth", method="lpbound_artifact"),
            64.6,
            0.05 / 64.6,
        ),
        (
            "CorrBound workload: block P95 over exact (3.9)",
            pick(cbs, "p95_over_truth", method="block_poly"),
            3.9,
            0.05 / 3.9,
        ),
        (
            "CorrBound workload: CorrBound-e P95 over exact (20)",
            pick(cbs, "p95_over_truth", method="corrbound_e"),
            20,
            0.5 / 20,
        ),
        (
            "CorrBound workload: LpBound P95 over exact (190)",
            pick(cbs, "p95_over_truth", method="lpbound_artifact"),
            190,
            0.5 / 190,
        ),
        (
            "CorrBound workload: block tighter than CorrBound-e (382)",
            pick(cbs, "n_block_poly_tighter", method="pairwise"),
            382,
        ),
        (
            "CorrBound workload: CorrBound-e tighter (18)",
            pick(cbs, "n_corrbound_e_tighter", method="pairwise"),
            18,
        ),
        (
            "CorrBound workload: median CorrBound-e / block (3.0)",
            pick(cbs, "median_corrbound_e_over_block_poly", method="pairwise"),
            3.0,
            0.05 / 3.0,
        ),
        (
            "CorrBound workload: max block / CorrBound-e (1.51)",
            pick(cbs, "max_block_poly_over_corrbound_e", method="pairwise"),
            1.51,
            0.005 / 1.51,
        ),
        (
            "CorrBound workload: block below exact (0)",
            pick(cbs, "n_below_truth", method="block_poly"),
            0,
        ),
        (
            "CorrBound workload: artifact counts below recount, median % (0.6)",
            100
            * (pick(cbs, "median_recount_over_artifact_truth", method="pairwise") - 1),
            0.6,
            0.05 / 0.6,
        ),
        (
            "CorrBound workload: artifact counts below recount, max % (39)",
            100 * (pick(cbs, "max_recount_over_artifact_truth", method="pairwise") - 1),
            39,
            0.5 / 39,
        ),
        (
            "CorrBound workload: stats seconds per query (25)",
            pick(cbs, "median_stats_s", method="pairwise"),
            25,
            0.5 / 25,
        ),
        (
            "CorrBound workload: solve seconds per query (0.38)",
            pick(cbs, "median_solve_s", method="pairwise"),
            0.38,
            0.005 / 0.38,
        ),
        (
            "CorrBound workload: dense queries (200)",
            pick(cbs, "n_dense", method="pairwise"),
            200,
        ),
        (
            "CorrBound workload: block tighter, dense (191)",
            pick(cbs, "n_block_poly_tighter_dense", method="pairwise"),
            191,
        ),
        (
            "CorrBound workload: block tighter, sparse (191)",
            pick(cbs, "n_block_poly_tighter_sparse", method="pairwise"),
            191,
        ),
        (
            "CorrBound workload: min edges (11)",
            pick(cbs, "min_edges", method="pairwise"),
            11,
        ),
        (
            "CorrBound workload: max edges (28)",
            pick(cbs, "max_edges", method="pairwise"),
            28,
        ),
        (
            "CorrBound workload: DBLP vertices, thousands (317)",
            pick(cbs, "dblp_vertices", method="pairwise") / 1000,
            317,
            0.5 / 317,
        ),
        (
            "CorrBound workload: DBLP edge rows, millions (2.1)",
            pick(cbs, "dblp_edge_rows", method="pairwise") / 1e6,
            2.1,
            0.05 / 2.1,
        ),
        (
            "cost table: hetionet free collect/LpBound (6.0)",
            pick(
                cai, "median_collect_over_lpbound", suite="hetionet", catalogue="free"
            ),
            6.0,
            0.05 / 6.0,
        ),
        (
            "cost table: hetionet free gain (608)",
            pick(cai, "median_gain_over_lpbound", suite="hetionet", catalogue="free"),
            608,
            0.5 / 608,
        ),
        (
            "cost table: hetionet free bound/exact (772)",
            pick(cai, "median_bound_over_truth", suite="hetionet", catalogue="free"),
            772,
            0.5 / 772,
        ),
        (
            "cost table: hetionet free_l2 collect/LpBound (6.8)",
            pick(
                cai,
                "median_collect_over_lpbound",
                suite="hetionet",
                catalogue="free_l2",
            ),
            6.8,
            0.05 / 6.8,
        ),
        (
            "cost table: hetionet free_l2 gain (1297)",
            pick(
                cai, "median_gain_over_lpbound", suite="hetionet", catalogue="free_l2"
            ),
            1297,
            0.5 / 1297,
        ),
        (
            "cost table: hetionet free_l2 bound/exact (340)",
            pick(cai, "median_bound_over_truth", suite="hetionet", catalogue="free_l2"),
            340,
            0.5 / 340,
        ),
        (
            "cost table: hetionet sparse_full collect/LpBound (8.2)",
            pick(
                cai,
                "median_collect_over_lpbound",
                suite="hetionet",
                catalogue="sparse_full",
            ),
            8.2,
            0.05 / 8.2,
        ),
        (
            "cost table: hetionet sparse_full gain (1724)",
            pick(
                cai,
                "median_gain_over_lpbound",
                suite="hetionet",
                catalogue="sparse_full",
            ),
            1724,
            0.5 / 1724,
        ),
        (
            "cost table: hetionet sparse_full bound/exact (211)",
            pick(
                cai,
                "median_bound_over_truth",
                suite="hetionet",
                catalogue="sparse_full",
            ),
            211,
            0.5 / 211,
        ),
        (
            "cost table: hetionet full collect/LpBound (54)",
            pick(
                cai, "median_collect_over_lpbound", suite="hetionet", catalogue="full"
            ),
            54,
            0.5 / 54,
        ),
        (
            "cost table: hetionet full gain (1724)",
            pick(cai, "median_gain_over_lpbound", suite="hetionet", catalogue="full"),
            1724,
            0.5 / 1724,
        ),
        (
            "cost table: hetionet full bound/exact (211)",
            pick(cai, "median_bound_over_truth", suite="hetionet", catalogue="full"),
            211,
            0.5 / 211,
        ),
        (
            "cost table: imdb free collect/LpBound (6.9)",
            pick(cai, "median_collect_over_lpbound", suite="imdb", catalogue="free"),
            6.9,
            0.05 / 6.9,
        ),
        (
            "cost table: imdb free gain (18.4)",
            pick(cai, "median_gain_over_lpbound", suite="imdb", catalogue="free"),
            18.4,
            0.5 / 18.4,
        ),
        (
            "cost table: imdb free bound/exact (7.7)",
            pick(cai, "median_bound_over_truth", suite="imdb", catalogue="free"),
            7.7,
            0.05 / 7.7,
        ),
        (
            "cost table: imdb free_l2 collect/LpBound (7.6)",
            pick(cai, "median_collect_over_lpbound", suite="imdb", catalogue="free_l2"),
            7.6,
            0.05 / 7.6,
        ),
        (
            "cost table: imdb free_l2 gain (20.9)",
            pick(cai, "median_gain_over_lpbound", suite="imdb", catalogue="free_l2"),
            20.9,
            0.5 / 20.9,
        ),
        (
            "cost table: imdb free_l2 bound/exact (7.3)",
            pick(cai, "median_bound_over_truth", suite="imdb", catalogue="free_l2"),
            7.3,
            0.05 / 7.3,
        ),
        (
            "cost table: imdb sparse_full collect/LpBound (8.3)",
            pick(
                cai,
                "median_collect_over_lpbound",
                suite="imdb",
                catalogue="sparse_full",
            ),
            8.3,
            0.05 / 8.3,
        ),
        (
            "cost table: imdb sparse_full gain (29.4)",
            pick(
                cai, "median_gain_over_lpbound", suite="imdb", catalogue="sparse_full"
            ),
            29.4,
            0.5 / 29.4,
        ),
        (
            "cost table: imdb sparse_full bound/exact (4.6)",
            pick(cai, "median_bound_over_truth", suite="imdb", catalogue="sparse_full"),
            4.6,
            0.05 / 4.6,
        ),
        (
            "cost table: imdb full collect/LpBound (25)",
            pick(cai, "median_collect_over_lpbound", suite="imdb", catalogue="full"),
            25,
            0.5 / 25,
        ),
        (
            "cost table: imdb full gain (29.4)",
            pick(cai, "median_gain_over_lpbound", suite="imdb", catalogue="full"),
            29.4,
            0.5 / 29.4,
        ),
        (
            "cost table: imdb full bound/exact (4.6)",
            pick(cai, "median_bound_over_truth", suite="imdb", catalogue="full"),
            4.6,
            0.05 / 4.6,
        ),
        (
            "cost table: hetionet lpbound bound/exact, 10^5 units (2.1)",
            1e-5
            * pick(
                cai, "median_bound_over_truth", suite="hetionet", catalogue="lpbound"
            ),
            2.1,
            0.05 / 2.1,
        ),
        (
            "cost table: imdb lpbound bound/exact (127)",
            pick(cai, "median_bound_over_truth", suite="imdb", catalogue="lpbound"),
            127,
            0.5 / 127,
        ),
        (
            "cost table: hetionet lpbound collect s (0.89)",
            pick(cai, "median_collect_s", suite="hetionet", catalogue="lpbound"),
            0.89,
            0.005 / 0.89,
        ),
        (
            "cost table: imdb lpbound collect s (0.52)",
            pick(cai, "median_collect_s", suite="imdb", catalogue="lpbound"),
            0.52,
            0.005 / 0.52,
        ),
        (
            "cost table: hetionet checked (115)",
            pick(cai, "n_checked", suite="hetionet", catalogue="full"),
            115,
        ),
        (
            "cost table: imdb checked (101)",
            pick(cai, "n_checked", suite="imdb", catalogue="full"),
            101,
        ),
        (
            "cost table: invalid bounds, all catalogs (0)",
            sum(float(r["n_invalid"]) for r in cai if r["catalogue"] != "checks"),
            0,
        ),
        (
            "cost text: hetionet sparse_full extra memory MB (60)",
            pick(
                cai,
                "median_peak_over_loaded_mb",
                suite="hetionet",
                catalogue="sparse_full",
            ),
            60,
            0.5 / 60,
        ),
        (
            "cost text: imdb sparse_full extra memory MB (44)",
            pick(
                cai, "median_peak_over_loaded_mb", suite="imdb", catalogue="sparse_full"
            ),
            44,
            0.5 / 44,
        ),
        (
            "cost text: hetionet full extra memory MB (626)",
            pick(cai, "median_peak_over_loaded_mb", suite="hetionet", catalogue="full"),
            626,
            0.5 / 626,
        ),
        (
            "cost text: imdb full extra memory MB (262)",
            pick(cai, "median_peak_over_loaded_mb", suite="imdb", catalogue="full"),
            262,
            0.5 / 262,
        ),
        (
            "cost ext: hetionet sparse_full max peak MB (429)",
            pick(cai, "max_peak_mb", suite="hetionet", catalogue="sparse_full"),
            429,
            0.5 / 429,
        ),
        (
            "cost ext: hetionet full max peak MB (6351)",
            pick(cai, "max_peak_mb", suite="hetionet", catalogue="full"),
            6351,
            0.5 / 6351,
        ),
        (
            "cost ext: imdb sparse_full max peak MB (341)",
            pick(cai, "max_peak_mb", suite="imdb", catalogue="sparse_full"),
            341,
            0.5 / 341,
        ),
        (
            "cost ext: imdb full max peak MB (4149)",
            pick(cai, "max_peak_mb", suite="imdb", catalogue="full"),
            4149,
            0.5 / 4149,
        ),
        (
            "cost text: polymatroid solve, fastest catalog median (0.08)",
            min(float(r["median_solve_s"]) for r in cai if r["catalogue"] != "checks"),
            0.08,
            0.005 / 0.08,
        ),
        (
            "cost text: polymatroid solve, slowest catalog median (0.10)",
            max(float(r["median_solve_s"]) for r in cai if r["catalogue"] != "checks"),
            0.1,
            0.005 / 0.1,
        ),
        (
            "cost text: flow solve hetionet (0.012)",
            pick(cai, "median_lpbound_flow_s", suite="hetionet", catalogue="checks"),
            0.012,
            0.0005 / 0.012,
        ),
        (
            "cost text: flow solve imdb (0.014)",
            pick(cai, "median_lpbound_flow_s", suite="imdb", catalogue="checks"),
            0.014,
            0.0005 / 0.014,
        ),
        (
            "cost: sparse_full equals full, max |dlog2| over both suites (0)",
            pick(
                cai,
                "max_abs_log2_sparse_full_minus_full",
                suite="hetionet",
                catalogue="checks",
            )
            + pick(
                cai,
                "max_abs_log2_sparse_full_minus_full",
                suite="imdb",
                catalogue="checks",
            ),
            0,
        ),
        (
            "cost ext: hetionet free_l2 share % (95)",
            100
            * pick(
                cai,
                "median_share_of_full_log_gain",
                suite="hetionet",
                catalogue="free_l2",
            ),
            95,
            0.5 / 95,
        ),
        (
            "cost ext: imdb free_l2 share % (90)",
            100
            * pick(
                cai, "median_share_of_full_log_gain", suite="imdb", catalogue="free_l2"
            ),
            90,
            0.5 / 90,
        ),
        (
            "earlier cost run: hetionet free collect/LpBound (6.0)",
            pick(
                cac, "median_collect_over_lpbound", suite="hetionet", catalogue="free"
            ),
            6.0,
            0.05 / 6.0,
        ),
        (
            "earlier cost run: imdb free collect/LpBound (6.9)",
            pick(cac, "median_collect_over_lpbound", suite="imdb", catalogue="free"),
            6.9,
            0.05 / 6.9,
        ),
        (
            "earlier cost run: hetionet free_l2 collect/LpBound (6.2)",
            pick(
                cac,
                "median_collect_over_lpbound",
                suite="hetionet",
                catalogue="free_l2",
            ),
            6.2,
            0.05 / 6.2,
        ),
        (
            "earlier cost run: imdb free_l2 collect/LpBound (7.1)",
            pick(cac, "median_collect_over_lpbound", suite="imdb", catalogue="free_l2"),
            7.1,
            0.05 / 7.1,
        ),
        (
            "earlier cost run: hetionet full collect/LpBound (52)",
            pick(
                cac, "median_collect_over_lpbound", suite="hetionet", catalogue="full"
            ),
            52,
            0.5 / 52,
        ),
        (
            "earlier cost run: imdb full collect/LpBound (24)",
            pick(cac, "median_collect_over_lpbound", suite="imdb", catalogue="full"),
            24,
            0.5 / 24,
        ),
        (
            "overlap ablation: hetionet median_gain_products (42)",
            pick(abo, "median_gain_products", suite="hetionet"),
            42,
            0.5 / 42,
        ),
        (
            "overlap ablation: imdb median_gain_products (2.0)",
            pick(abo, "median_gain_products", suite="imdb"),
            2.0,
            0.05 / 2.0,
        ),
        (
            "overlap ablation: hetionet median_full_over_products_gain (34)",
            pick(abo, "median_full_over_products_gain", suite="hetionet"),
            34,
            0.5 / 34,
        ),
        (
            "overlap ablation: imdb median_full_over_products_gain (14)",
            pick(abo, "median_full_over_products_gain", suite="imdb"),
            14,
            0.5 / 14,
        ),
        (
            "overlap ablation: hetionet median_gain_new_only (1674)",
            pick(abo, "median_gain_new_only", suite="hetionet"),
            1674,
            0.5 / 1674,
        ),
        (
            "overlap ablation: imdb median_gain_new_only (28.7)",
            pick(abo, "median_gain_new_only", suite="imdb"),
            28.7,
            0.05 / 28.7,
        ),
        (
            "overlap ablation: hetionet n_full_tighter_than_products (120)",
            pick(abo, "n_full_tighter_than_products", suite="hetionet"),
            120,
        ),
        (
            "overlap ablation: imdb n_full_tighter_than_products (120)",
            pick(abo, "n_full_tighter_than_products", suite="imdb"),
            120,
        ),
        (
            "overlap ablation: hetionet median_gain_size (15.1)",
            pick(abo, "median_gain_size", suite="hetionet"),
            15.1,
            0.05 / 15.1,
        ),
        (
            "overlap ablation: imdb median_gain_size (1.0)",
            pick(abo, "median_gain_size", suite="imdb"),
            1.0,
            0.005 / 1.0,
        ),
        (
            "overlap ablation: hetionet median_gain_one_side (608)",
            pick(abo, "median_gain_one_side", suite="hetionet"),
            608,
            0.5 / 608,
        ),
        (
            "overlap ablation: imdb median_gain_one_side (18.4)",
            pick(abo, "median_gain_one_side", suite="imdb"),
            18.4,
            0.05 / 18.4,
        ),
        (
            "overlap ablation: hetionet share beyond products % (48)",
            100 * pick(abo, "median_share_beyond_products", suite="hetionet"),
            48,
            0.5 / 48,
        ),
        (
            "overlap ablation: imdb share beyond products % (79)",
            100 * pick(abo, "median_share_beyond_products", suite="imdb"),
            79,
            0.5 / 79,
        ),
        (
            "robustness raw: all bound_slowdown_split_p95 (1.11)",
            pick(rrw, "bound_slowdown_split_p95", scope="all"),
            1.11,
            0.005 / 1.11,
        ),
        (
            "robustness raw: all duckdb_slowdown_split_p95 (5.26)",
            pick(rrw, "duckdb_slowdown_split_p95", scope="all"),
            5.26,
            0.005 / 5.26,
        ),
        (
            "robustness raw: cycle3 bound_over_duckdb_median (1.13)",
            pick(rrw, "bound_over_duckdb_median", scope="cycle3"),
            1.13,
            0.005 / 1.13,
        ),
        (
            "robustness raw: cycle4 bound_over_duckdb_median (1.78)",
            pick(rrw, "bound_over_duckdb_median", scope="cycle4"),
            1.78,
            0.005 / 1.78,
        ),
        (
            "robustness raw: diamond bound_over_duckdb_median (0.53)",
            pick(rrw, "bound_over_duckdb_median", scope="diamond"),
            0.53,
            0.005 / 0.53,
        ),
        (
            "robustness raw: k4 bound_over_duckdb_median (0.25)",
            pick(rrw, "bound_over_duckdb_median", scope="k4"),
            0.25,
            0.005 / 0.25,
        ),
        (
            "imdb sparse n1000 median collect s (9.3)",
            pick(iss, "median_collect_s", data="n1000"),
            9.3,
            0.05 / 9.3,
        ),
        (
            "imdb sparse n1000 max collect s (83)",
            pick(iss, "max_collect_s", data="n1000"),
            83,
            0.5 / 83,
        ),
        (
            "imdb sparse n1000 max peak GB (0.9)",
            pick(iss, "max_peak_mb", data="n1000") / 1024,
            0.9,
            0.05 / 0.9,
        ),
        (
            "imdb sparse growth in time, n1000/n200 (5.1)",
            pick(iss, "median_collect_s", data="n1000")
            / pick(iss, "median_collect_s", data="n200"),
            5.1,
            0.05 / 5.1,
        ),
        (
            "imdb sparse growth in rows, n1000/n200 (4.4)",
            pick(iss, "median_rows", data="n1000")
            / pick(iss, "median_rows", data="n200"),
            4.4,
            0.05 / 4.4,
        ),
        (
            "CorrBound artifact LP as shipped reproduces published, min (1.000000)",
            float(alp[0]["shipped_over_published_min"]),
            1.0,
            1e-06 / 1.0,
        ),
        (
            "CorrBound artifact LP as shipped reproduces published, max (1.000000)",
            float(alp[0]["shipped_over_published_max"]),
            1.0,
            1e-06 / 1.0,
        ),
        (
            "CorrBound Eq13/14 over shipped, median (2.4x)",
            float(alp[0]["median_eq_over_shipped"]),
            2.4,
            0.05 / 2.4,
        ),
        (
            "CorrBound Eq13/14 over shipped, min (1.5x)",
            float(alp[0]["min_eq_over_shipped"]),
            1.5,
            0.05 / 1.5,
        ),
        (
            "CorrBound Eq13/14 over shipped, max (6.1x)",
            float(alp[0]["max_eq_over_shipped"]),
            6.1,
            0.05 / 6.1,
        ),
        (
            "CorrBound Eq13/14 in artifact LP, median over exact (15.2x)",
            float(alp[0]["median_eq_over_truth"]),
            15.2,
            0.05 / 15.2,
        ),
        (
            "CorrBound shipped/Eq13/14 below exact (0)",
            int(alp[0]["n_shipped_below_truth"]) + int(alp[0]["n_eq_below_truth"]),
            0,
        ),
        (
            "CorrBound Eq13/14 in our program, median over exact (15.1x)",
            pick(cbc, "median_over_truth", catalog="corrbound"),
            15.1,
            0.05 / 15.1,
        ),
        (
            "block over Eq13/14 CorrBound-e, median (7.3x)",
            pick(cbc, "median_corrbound_over_block", catalog="checks"),
            7.3,
            0.05 / 7.3,
        ),
        (
            "combined tighter than block (28)/400",
            pick(cbc, "n_combined_tighter_than_block", catalog="checks"),
            28,
        ),
        (
            "combined at most 3.1% tighter (3.1)",
            100 * (pick(cbc, "max_block_over_combined", catalog="checks") - 1),
            3.1,
            0.05 / 3.1,
        ),
        (
            "combined median over exact (1.985)",
            pick(cbc, "median_over_truth", catalog="combined"),
            1.985,
            0.0005 / 1.985,
        ),
        (
            "block median over exact, combined run (1.987)",
            pick(cbc, "median_over_truth", catalog="block"),
            1.987,
            0.0005 / 1.987,
        ),
        (
            "combined run block equals sm_compare (0)",
            pick(cbc, "max_abs_log2_block_minus_compare", catalog="checks"),
            0,
        ),
        (
            "combined/corrbound below exact (0)",
            sum(int(r["n_below_truth"]) for r in cbc if r["catalog"] != "checks"),
            0,
        ),
        (
            "sweep k=14: cover over flow candidate (136x)",
            float(next(r for r in fsc if r["k"] == "14")["cover_bs2_bound"])
            / float(next(r for r in fsc if r["k"] == "14")["flow_bs2_bound"]),
            136,
            0.5 / 136,
        ),
        (
            "sweep k=14: polymatroid certificate seconds (1,425)",
            float(next(r for r in swc if r["k"] == "14")["poly_s"]),
            1425,
            0.5 / 1425,
        ),
        (
            "sweep k=12: polymatroid certificate seconds (20.1)",
            float(next(r for r in swc if r["k"] == "12")["poly_s"]),
            20.1,
            0.05 / 20.1,
        ),
        (
            "catalog norms-only size, KB (59)",
            max(int(r["catalogue_size"]) for r in cat) * 11 * 8 / 1000,
            59,
            0.5 / 59,
        ),
        (
            "G-CARE yago: LpBound/poly median where counts agree (12x)",
            statistics.median(
                float(r["entropic_p11"])
                / float(next(x for x in gpe if x["path"] == r["path"])["poly_bs2_p11"])
                for r in gce
                if r["dataset"] == "yago"
                and r.get("cover_bs1_p11") not in ("", None)
                and r["gcare_true"] == r["exact"]
            ),
            12,
            0.5 / 12,
        ),
        (
            "G-CARE yago: LpBound/poly median where counts differ (311x)",
            statistics.median(
                float(r["entropic_p11"])
                / float(next(x for x in gpe if x["path"] == r["path"])["poly_bs2_p11"])
                for r in gce
                if r["dataset"] == "yago"
                and r.get("cover_bs1_p11") not in ("", None)
                and r["gcare_true"] != r["exact"]
            ),
            311,
            0.5 / 311,
        ),
        (
            "VOI plan time over joining every block: hetionet (0.67x)",
            pick(vcs, "median_selected_over_full_time", suite="hetionet"),
            0.67,
            0.005 / 0.67,
        ),
        (
            "VOI plan cheaper than joining every block: hetionet (33)/40",
            sum(
                float(r["selected_over_full_time"]) < 1
                for r in vcr
                if r["suite"] == "hetionet"
            ),
            33,
        ),
        (
            "VOI plan time over joining every block: imdb (1.39x)",
            pick(vcs, "median_selected_over_full_time", suite="imdb"),
            1.39,
            0.005 / 1.39,
        ),
        (
            "predicates: rows passing year>=2005, median (46%)",
            100
            * statistics.median(
                float(r["rows_filtered_over_unfiltered"])
                for r in pbr
                if r["predicate"] == "year>=2005"
            ),
            46,
            0.5 / 46,
        ),
        (
            "predicates: rows passing year<1995, median (18%)",
            100
            * statistics.median(
                float(r["rows_filtered_over_unfiltered"])
                for r in pbr
                if r["predicate"] == "year<1995"
            ),
            18,
            0.5 / 18,
        ),
        (
            "predicates: year>=2005 catalog below LpBound (1.9x)",
            1 / pick(pbs, "median_catalog_over_lpbound", predicate="year>=2005"),
            1.9,
            0.05 / 1.9,
        ),
        (
            "predicates: year>=2005 catalog below LpBound on (40)/40",
            pick(pbs, "n_catalog_below_lpbound", predicate="year>=2005"),
            40,
        ),
        (
            "predicates: year<1995 catalog below LpBound (1.03x)",
            1 / pick(pbs, "median_catalog_over_lpbound", predicate="year<1995"),
            1.03,
            0.005 / 1.03,
        ),
        (
            "predicates: year<1995 catalog below LpBound on (22)/40",
            pick(pbs, "n_catalog_below_lpbound", predicate="year<1995"),
            22,
        ),
        (
            "predicates: per-predicate block stats below LpBound, low (4.2x)",
            min(1 / float(r["median_oracle_over_lpbound"]) for r in pbs),
            4.2,
            0.05 / 4.2,
        ),
        (
            "predicates: per-predicate block stats below LpBound, high (5.3x)",
            max(1 / float(r["median_oracle_over_lpbound"]) for r in pbs),
            5.3,
            0.05 / 5.3,
        ),
        (
            "predicates: bounds checked against exact counts (240)",
            3 * sum(int(r["n_checked"]) for r in pbs),
            240,
        ),
        ("predicates: invalid bounds (0)", sum(int(r["n_invalid"]) for r in pbs), 0),
        (
            "JOB-medium co-actor+co-keyword cycle-4 (1.94x)",
            pick(jfu, "lp_ratio", name="JOB Co-actor+Co-keyword (full JOB-medium)"),
            1.94,
            0.005,
        ),
        (
            "JOB-medium co-actor+co-company cycle-4 (2.80x)",
            pick(jfu, "lp_ratio", name="JOB Co-actor+Co-company (full JOB-medium)"),
            2.80,
            0.005,
        ),
        (
            "JOB-medium star control (1.00x)",
            pick(jfu, "lp_ratio", name="JOB Star-3 control (no cycle)"),
            1.00,
            0.0005,
        ),
        (
            "no improvement: IMDb replicated/co-director cycle-4 rows (8)",
            sum(
                1
                for r in jfu
                if ("cycle-4" in r["name"] and "JOB" not in r["name"])
                and float(r["lp_ratio"]) == 1.0
            ),
            8,
        ),
        (
            "certified: G-CARE queries checked (433)",
            sum(pick(gpc, "n", dataset=d) for d in ("aids", "yago")),
            433,
        ),
        (
            "certified: G-CARE flow below polymatroid (0)",
            sum(pick(gpc, "n_flow_below_poly", dataset=d) for d in ("aids", "yago")),
            0,
        ),
        (
            "certified: G-CARE flow reproduces gcare_eval exactly (0)",
            max(
                pick(gpc, "max_flow_vs_gcare_eval", dataset=d) for d in ("aids", "yago")
            ),
            0,
        ),
        (
            "poly headline: Hetionet |V|=6 below LpBound (236x)",
            1 / ph("hetionet", "6", "median_poly_over_entropic"),
            236,
            0.5 / 236,
        ),
        (
            "poly headline: Hetionet |V|=8 below LpBound (1720x)",
            1 / ph("hetionet", "8", "median_poly_over_entropic"),
            1720,
            0.005,
        ),
        (
            "poly headline: Hetionet |V|=10 below LpBound (10200x)",
            1 / ph("hetionet", "10", "median_poly_over_entropic"),
            10200,
            0.005,
        ),
        (
            "poly headline: IMDb |V|=6 below LpBound (9.7x)",
            1 / ph("imdb", "6", "median_poly_over_entropic"),
            9.7,
            0.05 / 9.7,
        ),
        (
            "poly headline: IMDb |V|=8 below LpBound (30x)",
            1 / ph("imdb", "8", "median_poly_over_entropic"),
            30,
            0.5 / 30,
        ),
        (
            "poly headline: IMDb |V|=10 below LpBound (87x)",
            1 / ph("imdb", "10", "median_poly_over_entropic"),
            87,
            0.5 / 87,
        ),
        (
            "Table long-suite polymatroid/entropic: hetionet |V|=6 (0.0042)",
            ph("hetionet", "6", "median_poly_over_entropic"),
            0.0042,
            5e-05 / 0.0042,
        ),
        (
            "Table long-suite polymatroid/entropic: hetionet |V|=8 (0.00058)",
            ph("hetionet", "8", "median_poly_over_entropic"),
            0.00058,
            5e-06 / 0.00058,
        ),
        (
            "Table long-suite polymatroid/entropic: hetionet |V|=10 (0.000098)",
            ph("hetionet", "10", "median_poly_over_entropic"),
            0.000098,
            5e-07 / 0.000098,
        ),
        (
            "Table long-suite polymatroid/entropic: imdb |V|=6 (0.10)",
            ph("imdb", "6", "median_poly_over_entropic"),
            0.10,
            0.005 / 0.10,
        ),
        (
            "Table long-suite polymatroid/entropic: imdb |V|=8 (0.034)",
            ph("imdb", "8", "median_poly_over_entropic"),
            0.034,
            0.0005 / 0.034,
        ),
        (
            "Table long-suite polymatroid/entropic: imdb |V|=10 (0.012)",
            ph("imdb", "10", "median_poly_over_entropic"),
            0.012,
            0.0005 / 0.012,
        ),
        (
            "poly headline: G-CARE aids below LpBound (15x)",
            1 / ph("gcare_aids", "all", "median_poly_over_entropic"),
            15,
            0.5 / 15,
        ),
        (
            "poly headline: G-CARE yago below LpBound (24x)",
            1 / ph("gcare_yago", "all", "median_poly_over_entropic"),
            24,
            0.5 / 24,
        ),
        (
            "poly program gap over cover bs2: hetionet |V|=6 (5.0x)",
            ph("hetionet", "6", "median_cover2_over_poly"),
            5.0,
            0.05 / 5.0,
        ),
        (
            "poly program gap over cover bs2: hetionet |V|=8 (6.7x)",
            ph("hetionet", "8", "median_cover2_over_poly"),
            6.7,
            0.05 / 6.7,
        ),
        (
            "poly program gap over cover bs2: hetionet |V|=10 (11.1x)",
            ph("hetionet", "10", "median_cover2_over_poly"),
            11.1,
            0.05 / 11.1,
        ),
        (
            "poly program gap over cover bs2: imdb |V|=6 (7.6x)",
            ph("imdb", "6", "median_cover2_over_poly"),
            7.6,
            0.05 / 7.6,
        ),
        (
            "poly program gap over cover bs2: imdb |V|=10 (11.6x)",
            ph("imdb", "10", "median_cover2_over_poly"),
            11.6,
            0.05 / 11.6,
        ),
        (
            "poly program gap over cover bs2: G-CARE aids (12x)",
            ph("gcare_aids", "all", "median_cover2_over_poly"),
            12,
            0.5 / 12,
        ),
        (
            "poly program gap over cover bs2: G-CARE yago (2.7x)",
            ph("gcare_yago", "all", "median_cover2_over_poly"),
            2.7,
            0.05 / 2.7,
        ),
        (
            "poly / exact: hetionet |V|=10 (2800)",
            ph("hetionet", "10", "median_poly_over_exact"),
            2800,
            0.05 / 2.8,
        ),
        (
            "poly / exact: imdb |V|=8 (6.4)",
            ph("imdb", "8", "median_poly_over_exact"),
            6.4,
            0.05 / 6.4,
        ),
        (
            "poly / exact: imdb |V|=6 (2.2)",
            ph("imdb", "6", "median_poly_over_exact"),
            2.2,
            0.05 / 2.2,
        ),
        (
            "poly / exact: imdb |V|=10 (15)",
            ph("imdb", "10", "median_poly_over_exact"),
            15,
            0.5 / 15,
        ),
        (
            "poly / exact: G-CARE aids (66)",
            ph("gcare_aids", "all", "median_poly_over_exact"),
            66,
            0.5 / 66,
        ),
        (
            "poly / exact: G-CARE yago (67)",
            ph("gcare_yago", "all", "median_poly_over_exact"),
            67,
            0.5 / 67,
        ),
        (
            "poly: invalid bounds against exact counts, all suites (0)",
            sum(float(r["n_poly_below_exact"]) for r in phl),
            0,
        ),
        (
            "poly below LpBound on Hetionet+IMDb (240)",
            sum(
                float(r["n_poly_below_entropic"])
                for r in phl
                if r["suite"] in ("hetionet", "imdb")
            ),
            240,
        ),
        (
            "poly below LpBound on short (109)",
            ph("short", "3-4", "n_poly_below_entropic"),
            109,
        ),
        (
            "poly YAGO below / equal LpBound (188)",
            ph("gcare_yago", "all", "n_poly_below_entropic"),
            188,
        ),
        (
            "poly YAGO equal to LpBound (4)",
            ph("gcare_yago", "all", "n_poly_eq_entropic"),
            4,
        ),
        (
            "flow within (2.41x) of polymatroid on every reported query, rounded up",
            math.ceil(
                max(
                    float(r["flow_over_poly_p11"])
                    for rs in (pce, gpe, spe, pse)
                    for r in rs
                )
                * 100
            )
            / 100,
            2.41,
        ),
        (
            "flow == polymatroid on (633) of the reported queries",
            sum(
                abs(float(r["flow_over_poly_p11"]) - 1) <= 1e-6
                for rs in (pce, gpe, spe, pse)
                for r in rs
            ),
            633,
        ),
        (
            "poly / exact: hetionet |V|=6 (47)",
            ph("hetionet", "6", "median_poly_over_exact"),
            47,
            0.5 / 47,
        ),
        (
            "poly / exact: hetionet |V|=8 (277)",
            ph("hetionet", "8", "median_poly_over_exact"),
            277,
            0.5 / 277,
        ),
        (
            "sweep certified: flow == polymatroid at k=12 and k=14 (2)",
            sum(
                r["status"] == "ok" and abs(float(r["flow_over_poly"]) - 1) <= 1e-6
                for r in swc
                if r["k"] in ("12", "14")
            ),
            2,
        ),
        (
            "spectral: certified Holder below polymatroid on (96)/240",
            pick(sps, "n_certified_below_poly", suite="all"),
            96,
        ),
        ("spectral: long-suite queries (240)", pick(sps, "n", suite="all"), 240),
        (
            "spectral: certified/poly median (1.11x)",
            pick(sps, "median_certified_over_poly", suite="all"),
            1.11,
            0.005 / 1.11,
        ),
        (
            "spectral: invalid certified + Lanczos bounds (0)",
            pick(sps, "n_invalid_certified", suite="all")
            + pick(sps, "n_invalid_lanczos", suite="all"),
            0,
        ),
        (
            "spectral: certified gain where tighter, median (1.25x)",
            pick(sps, "median_certified_gain_where_tighter", suite="all"),
            1.25,
            0.005 / 1.25,
        ),
        (
            "VOI: Hetionet top three blocks recover the full bound (median 1)",
            pick(vsl, "median_voi_frac_b3", suite="hetionet"),
            1,
            0.005,
        ),
        (
            "VOI: IMDb top two blocks recover the full bound (median 1)",
            pick(vsl, "median_voi_frac_b2", suite="imdb"),
            1,
            0.005,
        ),
        (
            "VOI: random three blocks recover about half on Hetionet (0.49)",
            pick(vsl, "median_random_frac_b3", suite="hetionet"),
            0.5,
            0.03,
        ),
        (
            "gap split: small random cycles (22)",
            sum(float(r["n"]) for r in osg),
            22,
        ),
        (
            "gap split: cycle-4 bs<=2 program gap at most (1.5x)",
            pick(osg, "median_program_gap_ub_bs2", k="4"),
            1.5,
            0.05 / 1.5,
        ),
        (
            "gap split: cycle-4 bs<=2 information gap at least (2.5x)",
            pick(osg, "median_info_gap_lb_bs2", k="4"),
            2.5,
            0.05 / 2.5,
        ),
        (
            "gap split: no search result exceeds its bound (0)",
            sum(float(r["n_lb_above_bound"]) for r in osg),
            0,
        ),
        (
            "spectral: Lanczos below polymatroid on (220)/240",
            pick(sps, "n_lanczos_below_poly", suite="all"),
            220,
        ),
        (
            "spectral: IMDb |V|=10 polymatroid/true (14.9x)",
            pick(sps, "median_poly_over_true", suite="imdb", n_vars="10"),
            14.9,
            0.05 / 14.9,
        ),
        (
            "spectral: IMDb |V|=10 min(Lanczos, poly)/true (3.2x)",
            pick(sps, "median_min_lanczos_poly_over_true", suite="imdb", n_vars="10"),
            3.2,
            0.05 / 3.2,
        ),
        (
            "sweep: 2^|V| program at k=14 takes (24) min",
            pick(swc, "poly_s", k="14") / 60.0,
            24,
            0.5 / 24,
        ),
        (
            "reported real-data queries (862)",
            len(pce) + len(gpe) + len(spe) + len(pse),
            862,
        ),
        (
            "poly solve, max over real data (63 s)",
            max(float(r["poly_s"]) for rs in (pce, gpe, spe, pse) for r in rs),
            63,
            0.5 / 63,
        ),
        (
            "poly solve median Hetionet |V|=6 (0.02 s)",
            statistics.median(
                float(r["poly_s"])
                for r in pce
                if r["suite"] == "hetionet" and r["n_vars"] == "6"
            ),
            0.02,
            0.005 / 0.02,
        ),
        (
            "poly solve median Hetionet |V|=8 (0.18 s)",
            statistics.median(
                float(r["poly_s"])
                for r in pce
                if r["suite"] == "hetionet" and r["n_vars"] == "8"
            ),
            0.18,
            0.005 / 0.18,
        ),
        (
            "poly solve median Hetionet |V|=10 (2.4 s)",
            statistics.median(
                float(r["poly_s"])
                for r in pce
                if r["suite"] == "hetionet" and r["n_vars"] == "10"
            ),
            2.4,
            0.05 / 2.4,
        ),
        (
            "bootstrap: poly rows crossing 1 (0)",
            sum(r["crosses_1"] == "True" for r in bci if r["ratio"].startswith("poly")),
            0,
        ),
        (
            "bootstrap: median ratios whose 95% CI crosses 1 (1)",
            sum(r["crosses_1"] == "True" for r in bci),
            1,
        ),
        (
            "bootstrap: AIDS cover bs2/entropic CI low (0.42)",
            pick(bci, "ci_lo", suite="gcare_aids", ratio="cover_bs2_p11/entropic_p11"),
            0.42,
            0.005 / 0.42,
        ),
        (
            "bootstrap: AIDS cover bs2/entropic CI high (1.00)",
            pick(bci, "ci_hi", suite="gcare_aids", ratio="cover_bs2_p11/entropic_p11"),
            1.0,
            0.005,
        ),
        ("certified: slice queries checked (80)", float(spc[0]["n"]), 80),
        (
            "certified: slice flow below polymatroid (0)",
            float(spc[0]["n_flow_below_poly"]),
            0,
        ),
        (
            "certified: slice flow reproduces slice_robustness exactly (0)",
            float(spc[0]["max_flow_vs_slice_robustness"]),
            0,
        ),
        (
            "certified: flow == polymatroid on all suites (633)",
            sum(
                pick(pcs, "n_flow_eq_poly", suite=x)
                for x in ("short", "hetionet", "imdb")
            )
            + sum(pick(gpc, "n_flow_eq_poly", dataset=d) for d in ("aids", "yago"))
            + float(spc[0]["n_flow_eq_poly"]),
            633,
        ),
        (
            "certified: flow reproduces pset_fair/p11_short exactly (0)",
            max(
                pick(pcs, "max_flow_vs_pset_fair", suite=x)
                for x in ("short", "hetionet", "imdb")
            ),
            0,
            0,
        ),
        (
            "robustness: synthetic cyclic queries timed (24)",
            pick(rb, "n", strategy="bound-guided"),
            24,
        ),
        (
            "cycle-6 p-values {1,inf} (17.6x)",
            pick(lpv, "improvement", p_values="1.0,inf"),
            17.6,
            0.05 / 17.6,
        ),
        (
            "cycle-6 p-values {1,2,inf} (35.7x)",
            pick(lpv, "improvement", p_values="1.0,2.0,inf"),
            35.7,
            0.05 / 35.7,
        ),
        (
            "cycle-6 p-values: l3, l5 add nothing (35.7x)",
            pick(lpv, "improvement", p_values="1.0,1.5,2.0,3.0,5.0,inf"),
            35.7,
            0.05 / 35.7,
        ),
        (
            "Hetionet suite bs2/bs1 collection cost, median (61x)",
            statistics.median(
                float(r["cover_bs2_s"]) / float(r["cover_bs1_s"]) for r in lse
            ),
            61,
            0.5 / 61,
        ),
        (
            "Hetionet suite bs2/bs1 collection cost, min (8x)",
            min(float(r["cover_bs2_s"]) / float(r["cover_bs1_s"]) for r in lse),
            8,
            0.5 / 8,
        ),
        (
            "Hetionet suite bs2/bs1 collection cost, max (558x)",
            max(float(r["cover_bs2_s"]) / float(r["cover_bs1_s"]) for r in lse),
            558,
            0.5 / 558,
        ),
        (
            "synthetic families bs2/bs1 collection cost, median (45x)",
            statistics.median(
                float(b["stats_ms"]) / float(a["stats_ms"])
                for a, b in zip(sco[0::2], sco[1::2], strict=True)
            ),
            45,
            0.5 / 45,
        ),
        (
            "G-CARE yago cyclic queries with a count (364)",
            sum(1 for r in gce if r["dataset"] == "yago" and r["exact"]),
            364,
            0,
        ),
        (
            "G-CARE yago counts differing from G-CARE's, all (80)",
            sum(
                1
                for r in gce
                if r["dataset"] == "yago"
                and r["exact"]
                and int(float(r["exact"])) != int(r["gcare_true"])
            ),
            80,
            0,
        ),
        (
            "G-CARE yago recount agrees with DuckDB (49/49)",
            sum(r["recount"] == r["duckdb_exact"] for r in gcr),
            49,
        ),
        ("G-CARE yago disputed queries recounted (49)", len(gcr), 49),
        (
            "G-CARE yago disputed queries in Petal families (45)",
            sum(r["petal"] == "True" for r in gcr),
            45,
        ),
        (
            "G-CARE yago Petal queries evaluated (100)",
            sum(
                "/Petal_" in r["path"]
                for r in gce
                if r["dataset"] == "yago" and r["cover_bs1_p11"]
            ),
            100,
        ),
        (
            "G-CARE aids cover-LP statistics gain (1.35x)",
            1 / gc("aids", "median_cover_bs2_over_entropic_p11"),
            1.35,
            0.005,
        ),
        (
            "G-CARE yago cover-LP statistics gain (8.9x)",
            1 / gc("yago", "median_cover_bs2_over_entropic_p11"),
            8.9,
            0.01,
        ),
        (
            "G-CARE aids cover gain (1.35x)",
            1 / gc("aids", "median_cover_bs2_over_entropic_p11"),
            1.35,
            0.005,
        ),
        (
            "G-CARE aids three-norm inference gap (3.4x)",
            gc("aids", "median_cover_bs1_over_entropic_p3"),
            3.4,
            0.05,
        ),
        (
            "G-CARE aids three-norm cover-LP statistics gain (21x)",
            1 / gc("aids", "median_cover_bs2_over_entropic_p3"),
            21,
            0.5,
        ),
        (
            "G-CARE yago exact==gcare among evaluated (143/192)",
            sum(
                r["gcare_true"] == r["exact"]
                for r in gce
                if r["dataset"] == "yago" and r.get("cover_bs1_p11") not in ("", None)
            ),
            143,
        ),
        (
            "G-CARE yago flow equal to entropic (4)",
            sum(
                float(r["flow_bs2_p11"]) == float(r["entropic_p11"])
                for r in gce
                if r["dataset"] == "yago" and r.get("cover_bs1_p11") not in ("", None)
            ),
            4,
        ),
        (
            "G-CARE yago: one flow bound below G-CARE truth (26 < 33)",
            sum(
                float(r["flow_bs2_p11"]) < float(r["gcare_true"])
                for r in gce
                if r["dataset"] == "yago" and r.get("cover_bs1_p11") not in ("", None)
            ),
            1,
        ),
        (
            "G-CARE: cyclic queries run (433 = 241 + 192)",
            gc("aids", "n_evaluated") + gc("yago", "n_evaluated"),
            433,
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
        tol = entry[3] if len(entry) > 3 else None
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

    absent = missing_from_paper(checks)
    print(
        f"  {len(checks) - len(absent)}/{len(checks)} pinned values occur in main.tex"
        " or the extended version"
    )
    for m in absent:
        print(f"    not in the paper: {m}")

    if failures:
        print("\nDRIFTED CLAIMS (paper text must be updated or CSV regenerated):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("  No drift detected.")


if __name__ == "__main__":
    main()
