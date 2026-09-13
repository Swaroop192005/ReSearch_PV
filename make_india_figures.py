"""
India figures for the paper, in the style of make_paper_figures.py (v3).
Reads the CSVs written by india_*.py; skips any figure whose CSV is missing.

Run:  python make_india_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"figure.constrained_layout.use": True, "axes.titlepad": 10,
                            "font.size": 11, "axes.titlesize": 12})
NAVY = "#1F4E79"; ORANGE = "#E67E22"; GREEN = "#2E7D32"; GREY = "#B0B7BF"
RES = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "figures" / "india"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"Calyxo_CdTe": "CdTe\n(AU)", "Kyocera_polySi": "poly-Si\n(AU)",
         "Trina_monoSi": "mono-Si\n(AU)", "NIST_Ground_monoSi": "mono-Si\n(US)"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("  wrote", name)


def have(name):
    ok = (RES / name).exists()
    if not ok:
        print(f"  [skip] {name} not found")
    return ok


def fig_gap():
    if not (have("india_matrix_raw.csv") and have("matrix_raw_power_nw.csv")):
        return
    M = pd.read_csv(RES / "india_matrix_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, wtag, title in zip(axes, ("nw", "w"), ("Irradiance only", "Irradiance + temperature")):
        ref = pd.read_csv(RES / f"matrix_raw_power_{wtag}.csv")
        auus = ref[ref["type"] == "cross-climate"]["gap"].mean()
        g = M[M.weather == wtag]
        srcs = list(SHORT)
        to = [g[(g.direction == "to-India") & (g.source == s)].gap.mean() for s in srcs]
        fr = [g[(g.direction == "from-India") & (g.target == s)].gap.mean() for s in srcs]
        x = np.arange(len(srcs)); w = 0.38
        ax.bar(x - w / 2, to, w, color=ORANGE, label="AU/US → India")
        ax.bar(x + w / 2, fr, w, color=NAVY, label="India → AU/US")
        ax.axhline(auus, color=GREEN, ls="--", lw=1.4, label="AU↔US cross-climate gap")
        ax.text(-0.62, auus - 0.002, f"AU↔US {auus:.4f}", ha="left", va="top",
                fontsize=9, color=GREEN)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in srcs])
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Generalization gap (R²)")
    axes[0].legend(loc="upper left", fontsize=9)
    save(fig, "fig11_india_gap.png")


def fig_recovery():
    if not have("india_recovery_raw.csv"):
        return
    R = pd.read_csv(RES / "india_recovery_raw.csv")
    g = R[(R.weather == "nw") & (R.source == "Trina_monoSi")]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for meth, lab, col, mk in (("gain", "scalar gain (1 parameter)", GREEN, "s"),
                               ("mlp", "MLP fine-tune", NAVY, "o"),
                               ("tree", "few-shot tree", GREY, "^")):
        s = g[g.method == meth].groupby("k").recovery.agg(["mean", "std"]) * 100
        ax.errorbar(s.index, s["mean"], yerr=s["std"], fmt=mk + "-", color=col, capsize=3, label=lab)
    ax.axhline(100, color=ORANGE, ls="--", lw=1.3); ax.axhline(0, color="#555", lw=1)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks(sorted(g.k.unique())); ax.set_xticklabels(sorted(g.k.unique()))
    ax.set_xlabel("Labelled Indian daylight hours (k)"); ax.set_ylabel("Gap recovered (%)")
    ax.set_title("Closing the AU → India gap"); ax.grid(alpha=0.3)
    fig.legend(loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    save(fig, "fig12_india_recovery.png")


def fig_err():
    if not have("india_error_by_irradiance.csv"):
        return
    E = pd.read_csv(RES / "india_error_by_irradiance.csv")
    g = E[(E.weather == "nw") & (E.source == "Trina_monoSi") & (E.band != "ALL")].copy()
    hi = g[g.band.isin(["800-1000", ">1000"])]
    if len(hi) == 2:
        pooled = lambda c: np.sqrt(np.sum(hi.n * hi[c] ** 2) / hi.n.sum())
        merged = dict(band=">800", n=int(hi.n.sum()), rmse_within=pooled("rmse_within"),
                      rmse_cross=pooled("rmse_cross"),
                      bias_cross=float(np.sum(hi.n * hi.bias_cross) / hi.n.sum()))
        g = pd.concat([g[~g.band.isin(hi.band)], pd.DataFrame([merged])], ignore_index=True)
    x = np.arange(len(g)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - w / 2, g.rmse_within, w, color=GREEN, label="within-site (India)")
    ax.bar(x + w / 2, g.rmse_cross, w, color=ORANGE, label="AU → India")
    for xi, n in zip(x, g.n):
        ax.text(xi, 0.002, f"n={n}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(g.band); ax.set_xlabel("Irradiance (W/m²)")
    ax.set_ylabel("RMSE (capacity factor)"); ax.set_title("Where the India error sits")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    save(fig, "fig13_india_err.png")


def fig_prior():
    if not have("india_native_raw.csv"):
        return
    H = pd.read_csv(RES / "india_native_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, wtag, title in zip(axes, ("nw", "w"), ("Irradiance only", "Irradiance + temperature")):
        g = H[H.weather == wtag]
        cols = [("zs_catboost", "zero-shot\nCatBoost"), ("zs_mlp", "zero-shot\nMLP"),
                ("mlp_ft20", "MLP fine-tune\n(20%)")]
        x = np.arange(len(cols)); w = 0.38
        for off, prior, col in ((-w / 2, "AU+US measured", ORANGE), (w / 2, "India-native PVGIS", NAVY)):
            p = g[g.prior == prior]
            ax.bar(x + off, [p[c].mean() for c, _ in cols], w, yerr=[p[c].std() for c, _ in cols],
                   capsize=3, color=col, label=prior + " prior")
        ax.axhline(g.within.mean(), color=GREEN, ls="--", lw=1.4, label="within-site ceiling")
        ax.text(len(cols) - 0.45, g.within.mean() + 0.002, f"ceiling {g.within.mean():.3f}",
                ha="right", va="bottom", fontsize=9, color=GREEN)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in cols]); ax.set_title(title)
        ax.set_ylim(0.85, 1.0); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("R² on held-out Indian plant\n(axis starts at 0.85)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    save(fig, "fig14_india_prior.png")


if __name__ == "__main__":
    for f in (fig_gap, fig_recovery, fig_err, fig_prior):
        f()
