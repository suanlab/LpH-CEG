#!/usr/bin/env bash
# Reproduce all LpH-CEG paper experiments and figures.
#
# Usage:
#   cd <repo root>
#   bash scripts/reproduce_all.sh          # run all (many hours; real-data suites need data/)
#   bash scripts/reproduce_all.sh quick    # tests+lint+figures (no LaTeX; the Docker default)
#   bash scripts/reproduce_all.sh smoke    # quick + paper build (~2 min, no long experiments)
#   bash scripts/reproduce_all.sh tests    # tests only
#   bash scripts/reproduce_all.sh lint     # ruff + mypy on contribution modules
#   bash scripts/reproduce_all.sh figures  # figures only
#   bash scripts/reproduce_all.sh paper    # build paper only
#
# Prerequisites: Python 3.10 (requirements.lock); LaTeX for the paper target
# Expected runtime of 'all': many hours (pset_fair, poly_certify, G-CARE dominate)

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

section() { echo -e "\n======== $1 ========"; }

# ---------------------------------------------------------------------------
run_tests() {
    section "1. Unit Tests"
    pytest tests/ -q --tb=short
}

# LpH-CEG-owned surface: contribution modules, all experiment scripts, and
# tests. The shared H-CEG core (src/core, src/utils, block_stats.py,
# pessimistic.py, wcoj/{evaluation,residual_bound,variable_order}) is
# maintained in the ../H-CEG project, so it is excluded from the lint gate.
LPH_SRC="src/estimators/lph_ceg.py src/estimators/lp_bound.py \
src/estimators/lpbound_baseline.py src/estimators/lpbound_entropic.py \
src/estimators/agm_bound.py src/estimators/flow_bound.py \
src/statistics/lp_stats.py src/statistics/pushdown_stats.py \
src/wcoj/lp_variable_order.py scripts/ tests/"

run_lint() {
    section "2. Lint & Type Check (LpH-CEG-owned: src modules + scripts + tests)"
    ruff format --check $LPH_SRC
    ruff check $LPH_SRC
    mypy --strict src/
}

run_experiments() {
    section "2b. Multi-Seed Synthetic Cycles (Table synthetic: multi_seed.csv)"
    python scripts/revision_experiments.py

    section "2c. JOB-medium, TPC-H and K-Hyperpath Exploration (bs=1 text results)"
    python scripts/job_experiment.py
    python scripts/tpch_experiment.py
    python scripts/explore_k_experiment.py

    section "3. Large-Scale 4-Way Comparison (Tables 1, 6, 7)"
    python scripts/large_scale_comparison.py

    section "4. Real Data Experiments (Tables 2, 5)"
    python scripts/lph_ceg_full_experiments.py

    section "5. JOB-Full Experiment (Table 5 update)"
    python scripts/job_full_experiment.py

    section "6. WCOJ End-to-End (Table 4)"
    python scripts/wcoj_e2e_experiment.py

    section "7. LP Scalability (Table 3)"
    python scripts/lp_scalability.py

    section "8. Cover-LP vs Entropic LpBound Crossover (Figure: crossover)"
    python scripts/scalability_vs_entropic.py

    section "9. End-to-End Plan Injection Robustness (Table/Figure: robust)"
    python scripts/e2e_plan_injection.py

    section "10. Hetionet Cyclic Real Benchmark (Table: realcycles)"
    python scripts/cyclic_real_benchmark.py

    section "11. Unified Baseline Comparison (Table: baselines)"
    python scripts/baseline_comparison.py

    section "12. Real-Workload Crossover on Hetionet (C1: cover vs entropic)"
    python scripts/crossover_real_workload.py

    section "13. WCOJ Simulator vs Kuzu Cross-Validation"
    python scripts/wcoj_engine_crossvalidate.py

    section "14. Diverse Real Subgraph Patterns (clique/chordal/diamond)"
    python scripts/subgraph_patterns_real.py

    section "15. Composition-Inequality Validity Search (C0 gate)"
    python scripts/validity_search.py --trials 15000   # matches the committed CSV and main.tex

    section "16. Skew/structure -> Tightness-Gap Characterization (C2)"
    python scripts/skew_gap_characterization.py

    section "17. WCOJ Ordering Study (C3: LpH vs baselines, paired stats)"
    python scripts/wcoj_ordering_study.py

    section "18. Broad Tail-Robustness Distribution (C4)"
    python scripts/robustness_broad.py
    python scripts/robustness_repeat.py      # same 24 queries, 5 shuffled timings per plan (run alone)
    python scripts/plan_ranking_study.py     # Kendall tau of bound rankings vs true cost

    section "19. p-value Sufficiency Ablation across Families"
    python scripts/pvalue_ablation.py

    section "20. LP-vs-Block Substitutes Ablation across Families"
    python scripts/substitutes_ablation.py

    section "20b. Step-5 Obligation over the Shannon Cone (C0 / SPOF)"
    python scripts/shannon_implication_check.py
    # every hyperpath, bs in {1,2,3}, |V| <= 6; 12 worker processes, ~40 min
    python scripts/shannon_check_wide.py

    section "20c. Pre-registered Query Suite (no bounds computed)"
    python scripts/register_query_suite.py

    section "20d. Evaluate Pre-registered Suite (377 queries, 2 workloads)"
    python scripts/evaluate_query_suite.py

    section "20e. Total Estimation Cost (statistics vs LP solve)"
    python scripts/stats_cost.py

    section "20e2. Cover LP vs. Valid-Weight Cutting Planes (refinement_hierarchy.csv)"
    python scripts/refinement_hierarchy.py

    section "20f. Statistics Hierarchy and the Statistics-vs-Inference 2x2"
    python scripts/statistics_hierarchy.py
    python scripts/statistics_vs_inference.py
    python scripts/real_frontier.py            # 109 short Hetionet cycles, 3K-row cap
    python scripts/real_frontier_truth.py      # exact counts of its 20 tightest bounds

    section "20g. Headline Cycle Table with the Statistics Axis"
    python scripts/headline_bs2.py

    # IMDb and G-CARE first: the long-suite scripts below (pset_fair, voi_selection, ...)
    # read the IMDb registry and data.
    section "20k. Second Real Dataset: IMDb Cycle Suite (needs JOB's IMDb CSVs; ~45 min)"
    python scripts/prepare_imdb_cycles.py     # derive typed relations from JOB's IMDb
    python scripts/register_imdb_suite.py     # pre-register 120 cycles (no bounds)
    python scripts/evaluate_imdb_suite.py     # all estimators + exact counts
    # 5x the films, |V|=6, bounds only (~1 h; peak 16 GB per worker)
    python scripts/prepare_imdb_cycles.py --n-movies 1000 --out data/imdb_cycles_n1000
    python scripts/evaluate_imdb_suite.py --data-dir data/imdb_cycles_n1000 \
        --tag imdb_scale_n1000 --no-exact --lengths 6
    python scripts/imdb_plan_injection.py     # bound-guided join order in DuckDB (~2 h)

    section "20l. External Benchmark: G-CARE Cyclic Queries (needs G-CARE data; ~2 h)"
    python scripts/gcare_prepare.py           # AIDS + YAGO into data/gcare/gcare.duckdb
    python scripts/gcare_suite.py             # 607 cyclic queries, ten and three norms
    python scripts/gcare_poly_certify.py      # flow >= polymatroid on every evaluated query (~8 h)
    python scripts/gcare_recount.py           # independent recount of the 49 disputed YAGO counts

    section "20h. Long-Cycle Suite, |V| in {6,8,10} (register, then evaluate; ~1.5 h)"
    python scripts/register_long_suite.py
    python scripts/evaluate_long_suite.py
    python scripts/agm_pairjoin_baseline.py   # AGM over pair-join sizes, same suite
    python scripts/stats_catalogue.py         # amortised statistics catalogue, same suite
    python scripts/lpbound_pset_check.py      # entropic baseline with p in {1..10, inf}
    python scripts/pset_fair.py               # both sides with p in {1..10, inf} (~2 h)
    python scripts/p11_reruns.py              # short suite, 2x2 and sweep with eleven norms
    python scripts/poly_certify.py            # flow >= polymatroid on every long-suite query (P11; ~5 h)
    python scripts/poly_certify.py --short    # ... and on the 109-query short suite
    python scripts/slice_robustness.py        # |V|=6 long suite on two random 20K samples (~3 h)
    python scripts/slice_poly_certify.py      # ... and certify those flow bounds by the polymatroid bound
    python scripts/spectral_bound.py          # trace-Holder spectral bounds on the long suites
    python scripts/opt_stats_gap.py           # program vs information gap on small cycles (OPT_S lower bounds)
    python scripts/voi_selection.py           # which pair joins to materialize: sketch-guided vs oracle vs random

    section "20i. Flow-LP Inference over Block Statistics (B3; ~1 h, suite part dominates)"
    python scripts/flow_validity_search.py    # 40 random shapes, all orders (~7 min)
    python scripts/flow_bound_blocks.py       # 2x2 grid, long suite, cycle sweep
    python scripts/sweep_poly_certify.py      # certify the k=12..16 sweep flow bounds (reads flow_scale.csv; long)

    section "20j. Join-Free Block Statistics and the Codegree l2 (B4; ~30 min)"
    python scripts/pushdown_stats_experiment.py   # families, sweep, long suite
    python scripts/corrbound_proxy.py             # entropic + pair-join sizes (~10 min)

    section "20m. Cost, Scale, Predicates and a Plan Guard (~1.5 h)"
    python scripts/voi_cost.py                # time of the sketch-guided plan vs joining every block (~30 min)
    python scripts/slice_scale.py             # |V|=6 long suite at a 30K-row prefix, cover vs LpBound (~40 min)
    # predicates need JOB's title.csv (production_year): set LPH_JOB_DIR as for prepare_imdb_cycles.py
    python scripts/predicate_blocks.py        # year predicates on IMDb |V|=6: predicate-free block catalog (~10 min)
    python scripts/guard_ordering.py          # bound-vs-engine-estimate plan guard (negative result; not in the paper)
    python scripts/ablation_p11_poly.py       # l2 ablation in the headline setting: eleven norms, polymatroid (~1 h)
    python scripts/cost_accuracy.py           # cost against tightness of four catalogs, one setting (Table V; ~45 min)
    # CorrBound's cyclic workload: needs a clone of https://github.com/christophmayerm/CorrBound
    # (set LPH_CORRBOUND_DIR; default ../external/CorrBound)
    python scripts/corrbound_sm_recount.py    # recount its 400 subgraph-matching queries with DuckDB (~5 min)
    python scripts/corrbound_sm_compare.py    # same queries, its norm set: LpBound check and block statistics (~25 min)

    # Aggregates last: they read the long-suite, IMDb and G-CARE results above.
    section "20n. Aggregates over Every Suite"
    python scripts/poly_headline.py           # headline medians with the bs<=2 polymatroid bound as the program
    python scripts/bootstrap_ci.py            # query-level bootstrap intervals of the headline medians
    python scripts/spectral_summary.py        # spectral bounds against the polymatroid bound and the truth
}

run_figures() {
    section "21. Generate Paper Figures (9 figures, all CSV-driven)"
    python scripts/generate_plots.py

    section "21b. Claim <-> CSV Audit"
    python scripts/claim_audit.py
}

run_paper() {
    section "22. Build Paper (pdflatex + bibtex)"
    cd docs/paper
    # `|| true`: let the integrity gate below report the diagnosis rather
    # than having `set -e` abort on pdflatex's exit code alone.
    pdflatex -interaction=nonstopmode main > /dev/null 2>&1 || true
    bibtex main > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main > /dev/null 2>&1 || true
    # extended version (full proofs), cited from the paper
    pdflatex -interaction=nonstopmode extended_version > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode extended_version > /dev/null 2>&1 || true
    if grep -q '^! ' extended_version.log; then
        echo "FAIL: LaTeX error(s) in extended_version.log"; exit 1
    fi

    # Build-integrity gate. A LaTeX *error* can silently delete a whole
    # environment from the PDF while page count, undefined refs and
    # overfull boxes all still look clean -- exactly how a missing
    # \begin{theorem} removed Theorem "Subsumption" for 12 commits.
    # main.log is gitignored and LaTeX may be absent (e.g. the Docker image
    # ships without it), in which case `grep -c` yields an empty string and
    # the gate would pass having verified nothing. Require the log to exist
    # and to be newer than the source before trusting it.
    if [ ! -f main.log ]; then
        echo "FAIL: main.log missing -- pdflatex did not run, so the paper"
        echo "      was NOT verified. Install LaTeX or use a target that"
        echo "      excludes run_paper."
        exit 1
    fi
    if [ main.tex -nt main.log ]; then
        echo "FAIL: main.log is older than main.tex -- stale build."
        exit 1
    fi
    errs=$(grep -c '^! ' main.log || true)
    errs=${errs:-0}
    if [ "$errs" -ne 0 ]; then
        echo "FAIL: $errs LaTeX error(s) in main.log:"
        grep -n '^! ' main.log | head
        exit 1
    fi
    # Environment balance: every \begin{X} needs its \end{X}.
    for env in theorem proof lemma definition proposition corollary; do
        b=$(grep -c "\\\\begin{$env}" main.tex || true)
        e=$(grep -c "\\\\end{$env}" main.tex || true)
        if [ "$b" -ne "$e" ]; then
            echo "FAIL: unbalanced $env environment (begin=$b end=$e)"
            exit 1
        fi
    done
    # ICDE page limit: the body (everything before the acknowledgment) must
    # end on page 12. Checking only where the acknowledgment starts is not
    # enough: body lines can spill onto page 13 above it.
    if command -v pdftotext > /dev/null; then
        # no early `exit` in awk: under pipefail a SIGPIPE to pdftotext
        # would abort the script without a message
        body_end=$(pdftotext main.pdf - | awk 'BEGIN{p=1} /\f/{p++} /^\f?A ?CKNOWLEDGMENT/ && !f {print p; f=1}')
        first13=$(pdftotext main.pdf - | awk 'BEGIN{p=1} /\f/{p++} p==13 && NF && !f {print; f=1}')
        if [ -z "$body_end" ] || [ "$body_end" -gt 13 ] || \
           { [ "$body_end" -eq 13 ] && [[ "$first13" != *CKNOWLEDGMENT* ]]; }; then
            echo "FAIL: body exceeds 12 pages (page 13 starts: '$first13')"
            exit 1
        fi
    fi
    echo "Paper build clean: 0 LaTeX errors, environments balanced, body <= 12 pages."
    cd ../..
}

# ---------------------------------------------------------------------------
case "${1:-all}" in
    tests)      run_tests ;;
    lint)       run_lint ;;
    experiments) run_experiments ;;
    figures)    run_figures ;;
    paper)      run_paper ;;
    quick)
        run_tests
        run_lint
        run_figures
        section "QUICK DONE — tests, lint, figures (no LaTeX needed)."
        ;;
    smoke)
        run_tests
        run_lint
        run_figures
        run_paper
        section "SMOKE DONE — tests, lint, figures, paper (no long experiments)."
        ;;
    all)
        run_tests
        run_lint
        run_experiments
        run_figures
        run_paper
        section "DONE — All experiments, figures, and paper reproduced."
        ;;
    *)
        echo "Usage: $0 {all|quick|smoke|tests|lint|experiments|figures|paper}"
        exit 1
        ;;
esac
