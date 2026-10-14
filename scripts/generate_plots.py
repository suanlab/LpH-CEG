"""Generate publication-quality plots for LpH-CEG paper.

Every figure reads its numbers from a results CSV; none embeds literals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("docs/paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IEEE publication style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (3.5, 2.8),
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        # IEEE PDF eXpress rejects Type 3 fonts; embed TrueType instead.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "ceg": "#6a3d9a",
    "lpbound": "#ff7f00",
    "hceg": "#cb181d",
    "lph": "#2171b5",
    "neutral": "#636363",
}

# ---------------------------------------------------------------------------
# Data source: figures read the 10-seed results CSV so they cannot drift
# from the recorded numbers.  Regenerate the CSV with the revision
# experiments, then re-run this script.
# ---------------------------------------------------------------------------
MULTISEED_CSV = Path("results/revision/multi_seed.csv")


def _load_multiseed() -> list[dict[str, str]]:
    """Load the 10-seed results CSV as a list of row dicts."""
    import csv

    with open(MULTISEED_CSV) as fh:
        return list(csv.DictReader(fh))


def _rows_for(experiment: str) -> list[dict[str, str]]:
    """Return rows of a given experiment, in file order."""
    return [r for r in _load_multiseed() if r["experiment"] == experiment]


def _read_csv(path: str) -> list[dict[str, str]]:
    """Load any results CSV as a list of row dicts, in file order."""
    import csv

    with open(Path(path)) as fh:
        return list(csv.DictReader(fh))


def plot_cycle_improvement() -> None:
    """Fig 1: Cycle-k vs improvement ratio (log scale), 10-seed mean±std."""
    rows = _rows_for("cycle_scale")
    k_values = [int(r["query"].split("_")[1]) for r in rows]
    lp_mean = [float(r["ratio_mean"]) for r in rows]
    lp_std = [float(r["ratio_std"]) for r in rows]

    fig, ax = plt.subplots()
    ax.semilogy(k_values, lp_mean, "o-", color=COLORS["lph"], label="LpH-CEG / CEG$_M$")
    ax.fill_between(
        k_values,
        [m - s for m, s in zip(lp_mean, lp_std, strict=False)],
        [m + s for m, s in zip(lp_mean, lp_std, strict=False)],
        alpha=0.15,
        color=COLORS["lph"],
    )

    for k, imp in zip(k_values, lp_mean, strict=False):
        if k in (4, 6, 8):
            ax.annotate(
                f"{imp:.0f}$\\times$",
                (k, imp),
                textcoords="offset points",
                xytext=(10, -2),
                fontsize=10,
                color=COLORS["lph"],
            )

    ax.set_xlabel("Cycle size $k$")
    ax.set_ylabel("Improvement ratio (log scale)")
    ax.set_xticks(k_values)
    ax.set_ylim(1, 2000)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left")
    fig.savefig(OUTPUT_DIR / "cycle_improvement.pdf")
    plt.close(fig)
    print("  Saved cycle_improvement.pdf")


def plot_overhead() -> None:
    """Fig 2: LP overhead vs cycle size (reads lp_scalability.csv)."""
    rows = _read_csv("results/scalability/lp_scalability.csv")
    k_values = [int(r["k"]) for r in rows]
    hceg_ms = [float(r["hceg_ms"]) for r in rows]
    lph_ms = [float(r["lph_ms"]) for r in rows]

    x = np.arange(len(k_values))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(
        x - width / 2, hceg_ms, width, label="H-CEG", color=COLORS["hceg"], alpha=0.85
    )
    ax.bar(
        x + width / 2, lph_ms, width, label="LpH-CEG", color=COLORS["lph"], alpha=0.85
    )

    ax.set_xlabel("Cycle size $k$")
    ax.set_ylabel("Estimation time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(loc="upper left")
    ax.set_ylim(0, 30)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.savefig(OUTPUT_DIR / "overhead.pdf")
    plt.close(fig)
    print("  Saved overhead.pdf")


def plot_pvalue_sensitivity() -> None:
    """Fig 3: p-set sufficiency across four query families.

    Reads results/ablation/pvalue_ablation.csv so the figure tracks the
    same 4-family measurement the text cites (not a single cycle-6 run).
    """
    csv_cols = [
        "{inf}",
        "{1,inf}",
        "{1,2,inf}",
        "{1,2,3,inf}",
        "{1,1.5,2,3,5,inf}",
    ]
    labels = [
        r"$\{\infty\}$",
        r"$\{1,\infty\}$",
        r"$\{1,2,\infty\}$",
        r"$\{1,2,3,\infty\}$",
        r"$\{1,1.5,2,3,5,\infty\}$",
    ]
    pretty = {
        "cycle4": "cycle-4",
        "cycle6": "cycle-6",
        "k4": r"$K_4$",
        "chordal5": "chordal-5",
    }
    rows = _read_csv("results/ablation/pvalue_ablation.csv")

    fig, ax = plt.subplots()
    markers = ["o", "s", "^", "D"]
    palette = [COLORS["lph"], COLORS["hceg"], COLORS["ceg"], COLORS["lpbound"]]
    for i, r in enumerate(rows):
        ax.plot(
            range(len(csv_cols)),
            [float(r[c]) for c in csv_cols],
            marker=markers[i % len(markers)],
            color=palette[i % len(palette)],
            label=pretty.get(r["family"], r["family"]),
        )

    ax.axvline(2, color="black", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r"$p$-value set $\mathcal{P}$")
    ax.set_ylabel("Improvement ratio")
    ax.set_yscale("log")
    ax.set_xticks(range(len(csv_cols)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax.annotate(
        "saturates",
        xy=(2, 30),
        xytext=(0.55, 26),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "black"},
        ha="center",
    )

    fig.savefig(OUTPUT_DIR / "pvalue_sensitivity.pdf")
    plt.close(fig)
    print("  Saved pvalue_sensitivity.pdf")


def plot_wcoj_comparison() -> None:
    """Fig 4: WCOJ iterations and time comparison."""
    queries = ["Hetionet\nCycle-4", "Synthetic\nCycle-4", "Triangle"]

    # Read the canonical hceg/lph_ceg rows straight from the results CSV so
    # the figure cannot drift.  Iterations are deterministic; times are
    # single-run wall-clock.
    rows = _read_csv("results/wcoj/wcoj_results.csv")
    keys = [
        "Hetionet Cycle-4 (heterogeneous)",
        "Cycle-4 (power-law, 100V, 400E)",
        "Triangle (power-law, 100V, 400E, control)",
    ]

    def _cell(query: str, strategy: str, col: str) -> float:
        for r in rows:
            if r["query"] == query and r["strategy"] == strategy:
                return float(r[col])
        raise KeyError(f"{query}/{strategy} missing from wcoj_results.csv")

    hceg_iters = [_cell(q, "hceg", "iterations") for q in keys]
    lph_iters = [_cell(q, "lph_ceg", "iterations") for q in keys]

    hceg_time = [_cell(q, "hceg", "time_ms") for q in keys]
    lph_time = [_cell(q, "lph_ceg", "time_ms") for q in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    x = np.arange(len(queries))
    width = 0.35

    ax1.bar(
        x - width / 2,
        [i / 1e6 for i in hceg_iters],
        width,
        label="H-CEG",
        color=COLORS["hceg"],
        alpha=0.85,
    )
    ax1.bar(
        x + width / 2,
        [i / 1e6 for i in lph_iters],
        width,
        label="LpH-CEG",
        color=COLORS["lph"],
        alpha=0.85,
    )
    ax1.set_ylabel("Iterations (millions)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, fontsize=9)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax1.set_title("(a) GenericJoin iterations")
    ax1.annotate(
        f"{hceg_iters[0] / lph_iters[0]:.1f}$\\times$ fewer",
        xy=(0.175, 1.3),
        fontsize=9,
        color=COLORS["lph"],
        fontweight="bold",
        ha="center",
    )

    ax2.bar(
        x - width / 2, hceg_time, width, label="H-CEG", color=COLORS["hceg"], alpha=0.85
    )
    ax2.bar(
        x + width / 2, lph_time, width, label="LpH-CEG", color=COLORS["lph"], alpha=0.85
    )
    ax2.set_ylabel("Execution time (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(queries, fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax2.set_title("(b) Wall-clock time (simulator)")
    ax2.annotate(
        f"{hceg_time[0] / lph_time[0]:.1f}$\\times$ faster",
        xy=(0.175, 700),
        fontsize=9,
        color=COLORS["lph"],
        fontweight="bold",
        ha="center",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wcoj_comparison.pdf")
    plt.close(fig)
    print("  Saved wcoj_comparison.pdf")


def plot_4way_comparison() -> None:
    """Fig 6: Data scalability on cycle-4, 10-seed mean with error band."""
    rows = _rows_for("data_scale")
    n_vertices = [int(r["n_vertices"]) for r in rows]
    ceg_mean = [float(r["ceg_mean"]) for r in rows]
    [float(r["ceg_std"]) for r in rows]
    lpb_mean = [float(r["lpbound_mean"]) for r in rows]
    lpb_std = [float(r["lpbound_std"]) for r in rows]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.loglog(
        n_vertices, ceg_mean, "s--", color=COLORS["ceg"], label="CEG$_M$", markersize=6
    )
    ax.loglog(
        n_vertices,
        lpb_mean,
        "o-",
        color=COLORS["lpbound"],
        label="LpBound",
        markersize=6,
    )
    ax.fill_between(
        n_vertices,
        [m - s for m, s in zip(lpb_mean, lpb_std, strict=False)],
        [m + s for m, s in zip(lpb_mean, lpb_std, strict=False)],
        alpha=0.15,
        color=COLORS["lpbound"],
    )

    for nv, c, lp in zip(n_vertices, ceg_mean, lpb_mean, strict=False):
        ratio = c / lp
        if nv in (200, 1000, 2000):
            ax.annotate(
                f"{ratio:.0f}$\\times$",
                (nv, lp),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                color=COLORS["lpbound"],
            )

    ax.set_xlabel("Number of vertices $|V|$")
    ax.set_ylabel("Upper bound (log scale)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    fig.savefig(OUTPUT_DIR / "4way_scalability.pdf")
    plt.close(fig)
    print("  Saved 4way_scalability.pdf")


def plot_skew_sensitivity() -> None:
    """Fig 7: Degree skew sensitivity on cycle-4, 10-seed mean±std."""
    rows = [r for r in _load_multiseed() if r["experiment"].startswith("skew_a=")]
    alphas = [float(r["experiment"].split("=")[1]) for r in rows]
    lp_mean = [float(r["ratio_mean"]) for r in rows]
    lp_std = [float(r["ratio_std"]) for r in rows]

    fig, ax = plt.subplots()
    ax.errorbar(
        alphas,
        lp_mean,
        yerr=lp_std,
        fmt="o-",
        color=COLORS["lph"],
        capsize=4,
        capthick=1.5,
        label="LP improvement (mean$\\pm$std)",
    )
    ax.fill_between(
        alphas,
        [m - s for m, s in zip(lp_mean, lp_std, strict=False)],
        [m + s for m, s in zip(lp_mean, lp_std, strict=False)],
        alpha=0.12,
        color=COLORS["lph"],
    )

    peak_idx = lp_mean.index(max(lp_mean))
    ax.annotate(
        f"peak: {lp_mean[peak_idx]:.1f}$\\pm${lp_std[peak_idx]:.1f}$\\times$",
        (alphas[peak_idx], lp_mean[peak_idx]),
        textcoords="offset points",
        xytext=(12, 5),
        fontsize=10,
        color=COLORS["lph"],
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["lph"]},
    )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Power-law exponent $\alpha$")
    ax.set_ylabel("LP improvement ratio")
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(OUTPUT_DIR / "skew_sensitivity.pdf")
    plt.close(fig)
    print("  Saved skew_sensitivity.pdf")


SCALE_ENTROPIC_CSV = Path("results/scalability/scale_entropic.csv")
FLOW_SCALE_CSV = Path("results/flow/flow_scale.csv")


def plot_crossover() -> None:
    """Fig 8: Cover LP vs entropic 2^|V| LpBound solve-time crossover.

    Reads ``results/scalability/scale_entropic.csv`` so the figure cannot
    drift from the recorded measurements.
    """
    import csv

    with open(SCALE_ENTROPIC_CSV) as fh:
        rows = list(csv.DictReader(fh))

    ks = [int(r["k"]) for r in rows]
    cover_ms = [float(r["cover_ms"]) for r in rows]
    # Entropic time only where it solved (status == ok); else intractable.
    ent_k = [int(r["k"]) for r in rows if r["status"] == "ok" and r["entropic_ms"]]
    ent_ms = [
        float(r["entropic_ms"])
        for r in rows
        if r["status"] == "ok" and r["entropic_ms"]
    ]
    intractable_ks = [int(r["k"]) for r in rows if r["status"] != "ok"]

    # Flow LP (Im et al. / LpBound's LPflow): polynomial inference at the
    # same bound as the 2^k program, at bs=1 and, over 4k orders, at bs=2.
    flow_k: list[int] = []
    flow1_ms: list[float] = []
    flow2_ms: list[float] = []
    if FLOW_SCALE_CSV.exists():
        with open(FLOW_SCALE_CSV) as fh:
            for r in csv.DictReader(fh):
                flow_k.append(int(r["k"]))
                flow1_ms.append(float(r["flow_bs1_ms"]))
                flow2_ms.append(float(r["flow_bs2_ms_total"]))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.semilogy(ks, cover_ms, "o-", color=COLORS["lph"], label="Cover LP ($3k$ vars)")
    if flow_k:
        ax.semilogy(
            flow_k,
            flow1_ms,
            "^-",
            color=COLORS["hceg"],
            label="Flow LP, $bs{=}1$ (LpBound's LPflow)",
        )
        ax.semilogy(
            flow_k,
            flow2_ms,
            "v-",
            color="tab:purple",
            label="Flow LP, $bs{=}2$ ($4k$ orders)",
        )
    ax.semilogy(
        ent_k,
        ent_ms,
        "s--",
        color=COLORS["lpbound"],
        label="Shannon-cone LP ($2^{k}$ vars)",
    )

    # Shade the region where the entropic LP is intractable.
    if intractable_ks:
        ax.axvspan(
            min(intractable_ks) - 0.5,
            max(ks) + 0.5,
            color=COLORS["hceg"],
            alpha=0.08,
        )
        ax.text(
            min(intractable_ks) + 0.1,
            ax.get_ylim()[1] * 0.3,
            "$2^k$ LP\nintractable",
            fontsize=8,
            color=COLORS["hceg"],
            va="top",
        )

    # Annotate the bound-tightness gap at a couple of representative k's.
    gap_by_k = {
        int(r["k"]): r["gap_cover_over_entropic"]
        for r in rows
        if r["gap_cover_over_entropic"]
    }
    for k_ann in (8, 12):
        if k_ann in gap_by_k:
            ax.annotate(
                f"gap {float(gap_by_k[k_ann]):.2f}$\\times$",
                (k_ann, dict(zip(ent_k, ent_ms, strict=False))[k_ann]),
                textcoords="offset points",
                xytext=(-10, 10) if k_ann == 8 else (6, -14),
                fontsize=8,
                color=COLORS["lpbound"],
            )

    ax.set_xlabel("Cycle size $k$ (= $|V|$)")
    ax.set_ylabel("LP solve time (ms, log scale)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.savefig(OUTPUT_DIR / "crossover_entropic.pdf")
    plt.close(fig)
    print("  Saved crossover_entropic.pdf")


PLAN_INJECTION_CSV = Path("results/e2e/plan_injection.csv")

# Compact x-axis labels for the robustness figure, keyed by full query name.
_ROBUST_LABELS = {
    "Cycle-4 (100V/400E)": "C4",
    "Cycle-5 (100V/400E)": "C5",
    "K4 4-clique (120V/800E)": "K4\n0.8K",
    "K4 4-clique (120V/1500E)": "K4\n1.5K",
    "K4 4-clique (120V/2500E)": "K4\n2.5K",
    "K4 4-clique (200V/2000E)": "K4\n200V",
    "Diamond / chordal-4 (120V/1500E)": "Diam",
    "Chordal 5-cycle (120V/1200E)": "Chord5",
    "Hetionet hetero-triangle (real)": "Het△",
}


def plot_robustness() -> None:
    """Fig 9: Per-query slowdown over best plan -- bound-guided vs engine.

    Reads ``results/e2e/plan_injection.csv``. The story is the *tail*: the
    engine optimizer spikes to 7-21x on dense 4-cliques, while bound-guided
    ordering stays within ~3.6x of optimal everywhere.
    """
    import csv

    with open(PLAN_INJECTION_CSV) as fh:
        rows = list(csv.DictReader(fh))

    labels = [_ROBUST_LABELS.get(r["query"], r["query"][:6]) for r in rows]
    bound = [float(r["bound_guided_slowdown"]) for r in rows]
    engine = [float(r["duckdb_slowdown"]) for r in rows]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(
        x - width / 2,
        bound,
        width,
        label="Bound-guided (ours)",
        color=COLORS["lph"],
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        engine,
        width,
        label="DuckDB optimizer",
        color=COLORS["hceg"],
        alpha=0.9,
    )
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Slowdown over best plan (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")

    bound_tail = max(bound)
    engine_tail = max(engine)
    ax.annotate(
        f"engine tail {engine_tail:.0f}$\\times$",
        xy=(bound.index(max(bound)), engine_tail),
        xytext=(0.5, engine_tail),
        fontsize=8,
        color=COLORS["hceg"],
        fontweight="bold",
    )
    ax.set_title(
        f"bound-guided tail {bound_tail:.1f}$\\times$ vs engine {engine_tail:.0f}$\\times$",
        fontsize=10,
    )
    fig.savefig(OUTPUT_DIR / "robustness.pdf")
    plt.close(fig)
    print("  Saved robustness.pdf")


SCALE_ENTROPIC_CSV_P = Path("results/scalability/scale_entropic.csv")
BASELINE_CSV_P = Path("results/baselines/baseline_comparison.csv")


def plot_positioning() -> None:
    """Figure 1: MEASURED cost-vs-looseness Pareto frontier.

    x = LP solve time (ms), y = looseness over the entropic optimum. Each
    point is a synthetic cycle-k query. Reads scale_entropic.csv (cover and
    entropic solve time + their gap) and baseline_comparison.csv (CEG_M
    looseness), so the frontier is measured, not schematic. The message:
    to reach the entropic optimum you pay a solve time that explodes with
    |V| in the 2^|V| formulation but not in LpBound's flow LP, which reaches
    the same bound in milliseconds (the inference axis at bs=1 is closed);
    the statistics axis (bs=2 block statistics, cover or flow LP) is what
    moves the bound below that optimum.
    """
    import csv

    with open(SCALE_ENTROPIC_CSV_P) as fh:
        se = [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    ks = [int(r["k"]) for r in se]
    cover_ms = [float(r["cover_ms"]) for r in se]
    ent_ms = [float(r["entropic_ms"]) for r in se]
    gap = [float(r["gap_cover_over_entropic"]) for r in se]

    # CEG_M looseness (ceg_m / entropic) from the baseline comparison, at the
    # same cheap cost tier as the cover LP (max-degree solve ~ cover solve).
    # Only plot points whose x-coordinate we actually measured; a fallback
    # constant here would put a fabricated coordinate on a figure whose
    # whole claim is that the frontier is measured.
    ceg_x, ceg_y, ceg_k = [], [], []
    with open(BASELINE_CSV_P) as fh:
        for r in csv.DictReader(fh):
            if r["query"].startswith("Cycle-"):
                k = int(r["query"].split("-")[1].split()[0])
                if k not in ks:
                    continue
                ceg_x.append(cover_ms[ks.index(k)])
                ceg_y.append(float(r["ceg_m"]) / float(r["entropic"]))
                ceg_k.append(k)

    # Flow LP (LpBound's polynomial LPflow) at bs=1 -- same bound as the
    # 2^|V| program (y=1) at ms cost -- and, on bs=2 block statistics, the
    # flow LP and the cover LP: the statistics axis, which leaves y=1
    # downwards.  Looseness is always relative to the entropic optimum on
    # single-relation statistics, so points below 1 are tighter than it.
    fl_x1, fl_x2, fl_y2, cv_x2, cv_y2, fl_k = [], [], [], [], [], []
    if FLOW_SCALE_CSV.exists():
        with open(FLOW_SCALE_CSV) as fh:
            for r in csv.DictReader(fh):
                k = int(r["k"])
                if k not in ks:
                    continue
                ent = float(r["entropic_bound"])
                fl_x1.append(float(r["flow_bs1_ms"]))
                fl_x2.append(float(r["flow_bs2_ms_total"]))
                fl_y2.append(float(r["flow_bs2_bound"]) / ent)
                cv_x2.append(float(r["cover_bs2_ms"]))
                cv_y2.append(float(r["cover_bs2_bound"]) / ent)
                fl_k.append(k)

    fig, ax = plt.subplots(figsize=(5.0, 4.1))

    # 2^|V| Shannon-cone formulation: tight (y=1) but solve time explodes.
    ax.plot(
        ent_ms,
        [1.0] * len(ent_ms),
        "o-",
        color=COLORS["lpbound"],
        markersize=5,
        label="Shannon-cone LP ($2^{|V|}$ vars), $bs{=}1$",
    )
    if fl_x1:
        ax.plot(
            fl_x1,
            [1.0] * len(fl_x1),
            "P-",
            color="tab:red",
            markersize=6,
            label="flow LP (LpBound's LPflow), $bs{=}1$: same bound",
        )
    _last_k = ks[-1]
    _last_s = ent_ms[-1] / 1000.0
    # plain units: matplotlib does not interpret LaTeX spacing macros
    _lab = f"{_last_s:.0f} s" if _last_s < 100 else f"{_last_s / 3600:.1f} h"
    ax.annotate(
        f"$k{{=}}{_last_k}$: {_lab}",
        (ent_ms[-1], 1.0),
        textcoords="offset points",
        xytext=(-6, 12),
        fontsize=8,
        ha="right",
        color=COLORS["lpbound"],
    )

    # cover frontier: cheap and near-tight, at every |V|.
    ax.plot(
        cover_ms,
        gap,
        "*-",
        color=COLORS["lph"],
        markersize=10,
        label="cover LP (ours): cheap, near-tight, decomposable",
    )
    ax.annotate(
        "$k{=}3$",
        (cover_ms[0], gap[0]),
        textcoords="offset points",
        xytext=(-4, -12),
        fontsize=8,
        color=COLORS["lph"],
    )
    ax.annotate(
        "$k{=}12$",
        (cover_ms[-1], gap[-1]),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
        color=COLORS["lph"],
    )

    # CEG_M: cheap but loose, looseness grows with |V|.
    ax.plot(
        ceg_x,
        ceg_y,
        "^--",
        color=COLORS["ceg"],
        markersize=6,
        label="CEG$_M$ / PANDA ($\\ell_\\infty$): cheap but loose",
    )

    # Statistics axis: the same two programs on bs=2 block statistics.
    if fl_x2:
        ax.plot(
            cv_x2,
            cv_y2,
            "*--",
            color=COLORS["lph"],
            markersize=8,
            markerfacecolor="white",
            label="cover LP on $bs{=}2$ block statistics",
        )
        ax.plot(
            fl_x2,
            fl_y2,
            "v-",
            color="tab:purple",
            markersize=6,
            label="flow LP on $bs{=}2$ block statistics ($4k$ orders)",
        )
        ax.annotate(
            f"$k{{=}}{fl_k[-1]}$",
            (fl_x2[-1], fl_y2[-1]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=8,
            color="tab:purple",
        )
        ax.axhline(1.0, color="gray", linewidth=0.6, alpha=0.5, zorder=0)
        ax.text(
            cv_x2[-1] * 4.0,
            0.75,
            "statistics axis:\nbelow the $bs{=}1$ optimum",
            fontsize=7.5,
            color="tab:purple",
            style="italic",
            va="top",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LP solve time (ms, log scale)")
    ax.set_ylabel("Bound / $bs{=}1$ entropic optimum")
    ax.grid(True, alpha=0.25, linestyle="--", which="both")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=7,
        framealpha=0.9,
        ncol=2,
    )
    fig.savefig(OUTPUT_DIR / "positioning.pdf")
    plt.close(fig)
    print("  Saved positioning.pdf")


def main() -> None:
    print("Generating LpH-CEG paper figures...")
    plot_positioning()
    plot_cycle_improvement()
    plot_overhead()
    plot_pvalue_sensitivity()
    plot_wcoj_comparison()
    plot_4way_comparison()
    plot_skew_sensitivity()
    plot_crossover()
    plot_robustness()
    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
