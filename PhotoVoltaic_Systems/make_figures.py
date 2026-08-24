"""
Generate the paper's figures from the results/ CSVs.

Reads whatever benchmark.py and transfer.py wrote and produces PNGs in figures/:
  fig1_degradation.png   - cross-site R2 drop per model and target
  fig2_rmse.png          - prediction error (RMSE): within-site vs cross-site
  fig3_recovery.png      - transfer-learning recovery vs target data used
  fig4_weather.png       - weather helps within-site, not cross-site

Usage:
    python make_figures.py            # uses the 'noweather' run for fig1/fig2
    python make_figures.py --tag weather

Robust to missing files: each figure is skipped (with a note) if its inputs
aren't present, so it works whether or not you've run the --weather experiments.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG_DIR = config.ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
RES = config.RESULTS_DIR
TASK = "power"
BLUE, ORANGE, GREEN, GREY = "#1F4E79", "#E67E22", "#2E7D32", "#888888"


def _load(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def _save(fig, name):
    fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig1_degradation(tag):
    deg = _load(f"degradation_{TASK}_{tag}.csv")
    if deg is None:
        print("  [skip] fig1: degradation file missing"); return
    deg = deg.copy()
    deg["target"] = deg["test"].str.split("->").str[-1]
    models = list(dict.fromkeys(deg["model"]))
    targets = list(dict.fromkeys(deg["target"]))
    x = np.arange(len(models)); w = 0.8 / max(len(targets), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, tgt in enumerate(targets):
        sub = deg[deg["target"] == tgt].set_index("model").reindex(models)
        ax.bar(x + i * w, sub["R2_drop"], w, label=f"→ {tgt}")
    ax.set_xticks(x + w * (len(targets) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("R² drop (within − cross)")
    ax.set_title("Cross-site degradation by model and target")
    ax.legend(title="Target site"); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig1_degradation.png")


def fig2_rmse(tag):
    o1 = _load(f"o1_{TASK}_{tag}.csv"); o2 = _load(f"o2_{TASK}_{tag}.csv")
    if o1 is None or o2 is None:
        print("  [skip] fig2: o1/o2 file missing"); return
    o2 = o2.copy()
    o2["source"] = o2["site"].str.split("->").str[0]
    o2["target"] = o2["site"].str.split("->").str[-1]
    source = o2["source"].iloc[0]
    targets = list(dict.fromkeys(o2["target"]))
    models = list(dict.fromkeys(o1["model"]))
    within = o1[o1["site"] == source].set_index("model")["RMSE"].reindex(models)
    x = np.arange(len(models)); n = 1 + len(targets); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, within, w, label=f"within-site ({source})", color=BLUE)
    for i, tgt in enumerate(targets, start=1):
        sub = o2[o2["target"] == tgt].set_index("model")["RMSE"].reindex(models)
        ax.bar(x + i * w, sub, w, label=f"cross-site → {tgt}")
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("RMSE (W)")
    ax.set_title("Prediction error: within-site vs cross-site")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig2_rmse.png")


def fig3_recovery():
    files = sorted(glob.glob(str(RES / f"o3_transfer_{TASK}_*_noweather.csv")))
    if not files:
        files = sorted(glob.glob(str(RES / f"o3_transfer_{TASK}_*.csv")))
    if not files:
        print("  [skip] fig3: no transfer files"); return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for f in files:
        df = pd.read_csv(f)
        target = os.path.basename(f).replace(f"o3_transfer_{TASK}_", "").replace("_noweather.csv", "").replace("_weather.csv", "").replace(".csv", "")
        fs = df[df["method"] == "few_shot"]
        ax.plot(fs["target_frac"] * 100, fs["recovery"] * 100, "o-", label=f"→ {target}")
    ax.axhline(0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("Target-site data used for adaptation (%)")
    ax.set_ylabel("Cross-site gap recovered (%)")
    ax.set_title("Few-shot transfer-learning recovery")
    ax.legend(title="Target site"); ax.grid(alpha=0.3)
    _save(fig, "fig3_recovery.png")


def fig4_weather():
    d0 = _load(f"degradation_{TASK}_noweather.csv")
    d1 = _load(f"degradation_{TASK}_weather.csv")
    o1n = _load(f"o1_{TASK}_noweather.csv"); o1w = _load(f"o1_{TASK}_weather.csv")
    o2n = _load(f"o2_{TASK}_noweather.csv"); o2w = _load(f"o2_{TASK}_weather.csv")
    if any(v is None for v in [d0, d1, o1n, o1w, o2n, o2w]):
        print("  [skip] fig4: need both weather and noweather runs"); return
    # average RMSE across boosting models for a compact comparison
    boost = ["CatBoost", "XGBoost", "HistGradientBoosting", "GradientBoosting", "RandomForest"]

    def mean_rmse(o1, o2):
        src = o2["site"].str.split("->").str[0].iloc[0]
        within = o1[(o1["site"] == src) & (o1["model"].isin(boost))]["RMSE"].mean()
        cross = o2[o2["model"].isin(boost)]["RMSE"].mean()
        return within, cross

    wn, cn = mean_rmse(o1n, o2n)
    ww, cw = mean_rmse(o1w, o2w)
    labels = ["Within-site", "Cross-site"]
    x = np.arange(2); w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, [wn, cn], w, label="no weather", color=GREY)
    ax.bar(x + w / 2, [ww, cw], w, label="with weather", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (W), avg of models")
    ax.set_title("Weather features help within-site, not cross-site")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for xi, (a, b) in zip(x, [(wn, ww), (cn, cw)]):
        ax.text(xi - w / 2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
    _save(fig, "fig4_weather.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="noweather", help="which run for fig1/fig2 (noweather|weather)")
    args = ap.parse_args()
    print(f"[figures] reading {RES}")
    fig1_degradation(args.tag)
    fig2_rmse(args.tag)
    fig3_recovery()
    fig4_weather()
    print(f"[done] figures in {FIG_DIR}")
