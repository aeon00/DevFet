"""
Aggregate the per-task CSVs produced by spangy_smoothing_analysis.py
and produce:

    1. smoothing_results_all.csv       — long-format master table
    2. smoothing_summary_table.csv     — mean ± SEM per smoothing level
    3. fig_afp_vs_smoothing.{pdf,png}  — AFP across smoothing levels
    4. fig_bands_vs_smoothing.{pdf,png}— B4/B5/B6 across smoothing levels

Both figures are styled for paper submission (300 dpi, vector PDF,
serif fonts, no top/right spines, colorblind-safe palette).
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "/envau/work/meca/users/dienye.h/B7_analysis/smoothing_effect/"
SMOOTHING_VALUES = [0, 5, 10, 20]


# ----------------------- matplotlib paper style --------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,    # embed TrueType — required by most journals
    "ps.fonttype": 42,
})


# ----------------------- load & aggregate --------------------------------

def load_all_results(results_dir=RESULTS_DIR):
    csvs = sorted(glob.glob(os.path.join(results_dir, "smoothing_results_task*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No per-task CSVs found in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df = df.dropna(subset=["afp", "B4", "B5", "B6"])
    return df


def summarise(df):
    """Mean and SEM at each smoothing level."""
    def sem(x):
        x = np.asarray(x, dtype=float)
        n = np.sum(np.isfinite(x))
        return np.nanstd(x, ddof=1) / np.sqrt(n) if n > 1 else np.nan

    agg = df.groupby("smoothing_iter").agg(
        n=("afp", "count"),
        afp_mean=("afp", "mean"), afp_sem=("afp", sem),
        B4_mean=("B4", "mean"),   B4_sem=("B4", sem),
        B5_mean=("B5", "mean"),   B5_sem=("B5", sem),
        B6_mean=("B6", "mean"),   B6_sem=("B6", sem),
    ).reset_index()
    return agg


# ----------------------- figures -----------------------------------------

def plot_afp(df, agg, out_stem):
    """AFP vs smoothing: thin per-subject trajectories + mean ± SEM."""
    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    # Per-subject (per-hemisphere) trajectories in light grey.
    for subj, sub_df in df.groupby("subject"):
        sub_df = sub_df.sort_values("smoothing_iter")
        ax.plot(
            sub_df["smoothing_iter"], sub_df["afp"],
            color="0.7", linewidth=0.5, alpha=0.5, zorder=1,
        )

    # Group mean ± SEM on top.
    ax.errorbar(
        agg["smoothing_iter"], agg["afp_mean"], yerr=agg["afp_sem"],
        marker="o", markersize=5.5, linewidth=1.6,
        color="#08519c", ecolor="#08519c",
        capsize=3, capthick=1.0, zorder=3,
        label=f"Mean ± SEM (n={int(agg['n'].iloc[0])})",
    )

    ax.set_xlabel("Laplacian smoothing iterations")
    ax.set_ylabel("Analyzed folding power (AFP)")
    ax.set_xticks(SMOOTHING_VALUES)
    ax.set_xlim(-1.5, max(SMOOTHING_VALUES) + 1.5)
    ax.margins(y=0.05)
    ax.legend(loc="best")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}")
    plt.close(fig)


def plot_bands(agg, out_stem):
    """B4 / B5 / B6 mean ± SEM vs smoothing iterations."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))

    # Colorblind-friendly (Okabe–Ito).
    palette = {"B4": "#0072B2", "B5": "#009E73", "B6": "#D55E00"}
    markers = {"B4": "o", "B5": "s", "B6": "^"}

    # Slight x-jitter so error bars don't overlap exactly.
    jitter = {"B4": -0.35, "B5": 0.0, "B6": 0.35}

    for band in ("B4", "B5", "B6"):
        x = agg["smoothing_iter"].values + jitter[band]
        ax.errorbar(
            x, agg[f"{band}_mean"], yerr=agg[f"{band}_sem"],
            marker=markers[band], markersize=5.5, linewidth=1.6,
            color=palette[band], ecolor=palette[band],
            capsize=3, capthick=1.0, label=band,
        )

    ax.set_xlabel("Laplacian smoothing iterations")
    ax.set_ylabel("Band power")
    ax.set_xticks(SMOOTHING_VALUES)
    ax.set_xlim(-1.5, max(SMOOTHING_VALUES) + 1.5)
    ax.margins(y=0.05)
    ax.legend(title="Frequency band", loc="best")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}")
    plt.close(fig)


# ----------------------- main --------------------------------------------

def main():
    df = load_all_results()
    print(f"Loaded {len(df)} rows across "
          f"{df['subject'].nunique()} subject-hemispheres "
          f"and {df['smoothing_iter'].nunique()} smoothing levels.")

    # Master long-format table.
    master_csv = os.path.join(RESULTS_DIR, "smoothing_results_all.csv")
    df.to_csv(master_csv, index=False)
    print(f"Wrote {master_csv}")

    # Summary table (the one to drop into a paper).
    agg = summarise(df)
    summary_csv = os.path.join(RESULTS_DIR, "smoothing_summary_table.csv")
    agg.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")
    print("\nSummary (mean ± SEM):")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(agg.to_string(index=False))

    # Figures.
    plot_afp(df, agg, os.path.join(RESULTS_DIR, "fig_afp_vs_smoothing"))
    plot_bands(agg, os.path.join(RESULTS_DIR, "fig_bands_vs_smoothing"))
    print("\nFigures written (.pdf + .png).")


if __name__ == "__main__":
    main()