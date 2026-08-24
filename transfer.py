"""
O3 - Transfer learning / domain adaptation
==========================================

This is the novel core of the project. After O2 shows that a model trained on
one site LOSES accuracy on an unseen site, this module tries to RECOVER that
accuracy and reports how much of the gap is closed.

For a fair, comparable evaluation, the target site is split ONCE into:
  * target-test  (held out, used for EVERY score below)
  * target-pool  (the rest; the few-shot samples are drawn from here)

Reference points computed on target-test:
  * zero_shot      : source-only model (this is the O2 result) -- lower bound
  * within_target  : model trained on the full target-pool      -- upper bound

Two adaptation methods (the "2 variants" from the plan):
  1. few_shot  : pool source + k% of target-pool, retrain. Needs a few target
                 labels. Reported for k in --fractions (e.g. 5%, 10%, 20%).
  2. coral     : CORAL feature alignment (unsupervised -- NO target labels).
                 Aligns source feature covariance to the target, then applies
                 the source model. numpy-only implementation.

Headline number:
  recovery_ratio = (adapted - zero_shot) / (within_target - zero_shot)
  i.e. fraction of the lost accuracy that the method wins back (1.0 = fully
  recovered, 0.0 = no better than zero-shot).

Usage:
    python transfer.py --task power --target B --fractions 0.05,0.1,0.2
    python transfer.py --task power --target B --weather
    python transfer.py --task fault --target B
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import config
import data as data_mod
import models as model_mod
import metrics as metric_mod

warnings.filterwarnings("ignore")

KEY = {"power": "R2", "fault": "F1"}


# --------------------------------------------------------------------------
# model selection: use the strongest available model for the task
# --------------------------------------------------------------------------
def pick_model(task):
    zoo = model_mod.get_regressors() if task == "power" else model_mod.get_classifiers()
    for name in ["CatBoost", "XGBoost", "HistGradientBoosting", "GradientBoosting", "RandomForest"]:
        if name in zoo:
            return name, zoo[name]
    name = next(iter(zoo))
    return name, zoo[name]


def score(model, X_tr, y_tr, X_te, y_te, task):
    m = clone(model) if hasattr(model, "get_params") else model
    m.fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    if task == "fault":
        proba = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else None
        return metric_mod.classification_metrics(y_te, y_pred, proba)[KEY[task]]
    return metric_mod.regression_metrics(y_te, y_pred)[KEY[task]]


# --------------------------------------------------------------------------
# CORAL feature alignment  (Sun, Feng & Saenko, 2016) -- numpy only
# --------------------------------------------------------------------------
def _mat_pow(M, p, eps=1e-6):
    M = M + eps * np.eye(M.shape[0])
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    return (V * (w ** p)) @ V.T


def coral_align(Xs, Xt):
    """Return source features re-colored to match the target covariance.
    Both are z-normalised by source stats first."""
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-8
    Xs_z, Xt_z = (Xs - mu) / sd, (Xt - mu) / sd
    Cs = np.cov(Xs_z, rowvar=False)
    Ct = np.cov(Xt_z, rowvar=False)
    A = _mat_pow(Cs, -0.5) @ _mat_pow(Ct, 0.5)
    return Xs_z @ A, Xt_z  # aligned source, normalised target


# --------------------------------------------------------------------------
# main experiment
# --------------------------------------------------------------------------
def run(df, task, source_sites, target_site, fractions, use_weather, target_test_size=0.4):
    name, model = pick_model(task)
    feats = data_mod.feature_columns(use_weather)

    # source data
    Xs, ys = data_mod.get_xy(data_mod.site_frame(df, source_sites), task, use_weather)
    # target data -> fixed test split + adaptation pool
    tgt = data_mod.site_frame(df, [target_site])
    Xt, yt = data_mod.get_xy(tgt, task, use_weather)
    strat = yt if task == "fault" else None
    Xt_pool, Xt_test, yt_pool, yt_test = train_test_split(
        Xt, yt, test_size=target_test_size, random_state=config.RANDOM_STATE, stratify=strat)

    # references
    zero_shot = score(model, Xs, ys, Xt_test, yt_test, task)
    within = score(model, Xt_pool, yt_pool, Xt_test, yt_test, task)
    gap = within - zero_shot

    rows = [
        {"method": "zero_shot (O2)", "target_frac": 0.0, KEY[task]: round(zero_shot, 4), "recovery": 0.0},
    ]

    # method 1: few-shot fine-tuning (pool source + k% target)
    rng = np.random.default_rng(config.RANDOM_STATE)
    n_pool = len(Xt_pool)
    for frac in fractions:
        k = max(5, int(round(frac * len(Xt))))
        k = min(k, n_pool)
        idx = rng.choice(n_pool, size=k, replace=False)
        Xk, yk = Xt_pool.iloc[idx], yt_pool[idx]
        X_comb = pd.concat([Xs, Xk], ignore_index=True)
        y_comb = np.concatenate([ys, yk])
        s = score(model, X_comb, y_comb, Xt_test, yt_test, task)
        rec = (s - zero_shot) / gap if abs(gap) > 1e-9 else float("nan")
        rows.append({"method": "few_shot", "target_frac": round(frac, 3),
                     KEY[task]: round(s, 4), "recovery": round(rec, 3)})

    # method 2: CORAL (unsupervised, no target labels)
    Xs_al, Xt_test_z = coral_align(Xs.values, Xt_test.values)
    s = score(model, pd.DataFrame(Xs_al, columns=feats), ys,
              pd.DataFrame(Xt_test_z, columns=feats), yt_test, task)
    rec = (s - zero_shot) / gap if abs(gap) > 1e-9 else float("nan")
    rows.append({"method": "coral (no labels)", "target_frac": 0.0,
                 KEY[task]: round(s, 4), "recovery": round(rec, 3)})

    rows.append({"method": "within_target (upper bound)", "target_frac": 1.0,
                 KEY[task]: round(within, 4), "recovery": 1.0})

    res = pd.DataFrame(rows)
    return name, zero_shot, within, res


def _print(title, df):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    with pd.option_context("display.width", 120, "display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="O3 transfer learning / domain adaptation")
    ap.add_argument("--task", choices=["power", "fault"], default="power")
    ap.add_argument("--target", default=None, help="target site (default: config.TEST_SITES[0])")
    ap.add_argument("--source", default=None, help="comma-sep source sites (default: config.TRAIN_SITES)")
    ap.add_argument("--fractions", default="0.05,0.1,0.2", help="few-shot target fractions")
    ap.add_argument("--weather", action="store_true", help="include weather features")
    args = ap.parse_args()

    df = data_mod.load_data()
    source = args.source.split(",") if args.source else config.TRAIN_SITES
    target = args.target or config.TEST_SITES[0]
    fractions = [float(x) for x in args.fractions.split(",")]

    print(model_mod.availability_note())
    print(f"[transfer] task={args.task}  source={source} -> target={target}  "
          f"weather={'ON' if args.weather else 'OFF'}")

    name, zero_shot, within, res = run(df, args.task, source, target, fractions, args.weather)
    metric = KEY[args.task]
    _print(f"O3 transfer learning | task={args.task} | model={name} | "
           f"source={'+'.join(source)} -> {target}", res)

    best = res[res.method.isin(["few_shot", "coral (no labels)"])]
    best_row = best.loc[best[metric].idxmax()]
    print(f"\nHEADLINE: zero-shot {metric}={zero_shot:.4f}  ->  best adapted "
          f"{metric}={best_row[metric]:.4f}  (recovery {best_row['recovery']:.0%} of the "
          f"{within-zero_shot:.4f} gap)  via {best_row['method']}.")

    tag = "weather" if args.weather else "noweather"
    out = config.RESULTS_DIR / f"o3_transfer_{args.task}_{target}_{tag}.csv"
    res.to_csv(out, index=False)
    print(f"[done] written to {out}")


if __name__ == "__main__":
    main()
