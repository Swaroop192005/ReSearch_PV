"""
matrix_experiment.py -- all-pairs cross-site generalization matrix with rigor.

For every ordered (source -> target) pair among the loaded sites, and for the
within-site diagonal, it trains a model and records R2, repeated over several
random seeds to give mean +/- std and 95% confidence intervals. It then groups
the off-diagonal gaps into cross-technology (same location) vs cross-climate
(different location) and runs a significance test on the difference.

Usage:
  python matrix_experiment.py --task power            # irradiance only
  python matrix_experiment.py --task power --weather  # + ambient temperature
"""
import argparse, itertools, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from scipy import stats

import config, data as d, metrics as met, models as m
warnings.filterwarnings("ignore")

SEEDS = [42, 1, 2, 7, 123]


def site_meta(s):
    loc = "NIST" if "nist" in s.lower() else "Alice"
    sl = s.lower()
    tech = "mono-Si" if "mono" in sl else ("poly-Si" if "poly" in sl else ("CdTe" if "cdte" in sl else "?"))
    return loc, tech


def pick_model_name(task):
    zoo = m.get_regressors() if task == "power" else m.get_classifiers()
    for n in ["CatBoost", "XGBoost", "HistGradientBoosting", "GradientBoosting", "RandomForest"]:
        if n in zoo:
            return n, zoo[n]
    n = next(iter(zoo)); return n, zoo[n]


def score(model, Xtr, ytr, Xte, yte, task):
    mm = clone(model)
    try: mm.set_params(random_state=0)
    except Exception: pass
    mm.fit(Xtr, ytr); yp = mm.predict(Xte)
    key = "R2" if task == "power" else "F1"
    if task == "fault":
        pr = mm.predict_proba(Xte)[:, 1] if hasattr(mm, "predict_proba") else None
        return met.classification_metrics(yte, yp, pr)[key]
    return met.regression_metrics(yte, yp)[key]


def with_seed(base, seed):
    mm = clone(base)
    try: mm.set_params(random_state=seed)
    except Exception: pass
    return mm


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return (x.mean() - h, x.mean() + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["power", "fault"], default="power")
    ap.add_argument("--weather", action="store_true")
    args = ap.parse_args()

    df = d.load_data()
    sites = sorted(df[config.SITE_COL].unique())
    name, base = pick_model_name(args.task)
    key = "R2" if args.task == "power" else "F1"
    print(f"[matrix] model={name}  sites={sites}  weather={'ON' if args.weather else 'OFF'}  seeds={SEEDS}")

    rec = []
    for seed in SEEDS:
        within = {}
        for s in sites:
            X, y = d.get_xy(d.site_frame(df, [s]), args.task, args.weather)
            strat = y if args.task == "fault" else None
            xa, xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
            r = score(with_seed(base, seed), xa, ya, xb, yb, args.task)
            within[s] = r
            rec.append({"seed": seed, "source": s, "target": s, "type": "within", key: r, "gap": 0.0})
        for src, tgt in itertools.permutations(sites, 2):
            Xs, ys = d.get_xy(d.site_frame(df, [src]), args.task, args.weather)
            Xt, yt = d.get_xy(d.site_frame(df, [tgt]), args.task, args.weather)
            r = score(with_seed(base, seed), Xs, ys, Xt, yt, args.task)
            ls, _ = site_meta(src); lt, _ = site_meta(tgt)
            typ = "cross-climate" if ls != lt else "cross-technology"
            rec.append({"seed": seed, "source": src, "target": tgt, "type": typ, key: r, "gap": within[tgt] - r})

    R = pd.DataFrame(rec)

    # ---- mean matrix (rows=source, cols=target) ----
    piv = R.groupby(["source", "target"])[key].mean().unstack().reindex(index=sites, columns=sites)
    print(f"\n===== MEAN {key} MATRIX (rows=train/source, cols=test/target; diagonal=within-site) =====")
    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:.3f}"):
        print(piv.round(3))
    piv.round(4).to_csv(config.RESULTS_DIR / f"matrix_{args.task}_{'w' if args.weather else 'nw'}.csv")

    # ---- gap summary by type, with CI ----
    print(f"\n===== GENERALIZATION GAP by shift type (mean {key} drop, 95% CI over {len(SEEDS)} seeds x pairs) =====")
    for typ in ["cross-technology", "cross-climate"]:
        g = R[R.type == typ]["gap"].values
        lo, hi = ci95(g)
        print(f"  {typ:18s}: n={len(g):2d}  mean={g.mean():.4f}  std={g.std(ddof=1):.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")

    # ---- significance test ----
    tech = R[R.type == "cross-technology"]["gap"].values
    clim = R[R.type == "cross-climate"]["gap"].values
    if len(tech) and len(clim):
        U, p_u = stats.mannwhitneyu(clim, tech, alternative="greater")
        tt, p_t = stats.ttest_ind(clim, tech, equal_var=False)
        print(f"\n===== SIGNIFICANCE: cross-climate gap > cross-technology gap? =====")
        print(f"  Mann-Whitney U={U:.1f}, one-sided p={p_u:.2e}")
        print(f"  Welch t={tt:.2f}, two-sided p={p_t:.2e}")
        print(f"  -> ratio of mean gaps (climate/technology) = {clim.mean()/tech.mean():.1f}x")

    R.to_csv(config.RESULTS_DIR / f"matrix_raw_{args.task}_{'w' if args.weather else 'nw'}.csv", index=False)
    print(f"\n[done] matrices + raw records in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
