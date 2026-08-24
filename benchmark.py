"""
Benchmark runner.

  O1  within-site   : train & test each model on the SAME site (reproduces the
                      high accuracy reported in prior work -> establishes credibility)
  O2  cross-site    : train on TRAIN_SITES, test zero-shot on TEST_SITES, and
                      report the DEGRADATION (delta) vs within-site -> the paper's
                      core evidence. This is scaffolding for O3 (transfer.py).

Usage:
    python benchmark.py --task power        # or --task fault  or  --task both
    python benchmark.py --task both --weather
    python benchmark.py --experiment o1     # o1 | o2 | both  (default both)
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import config
import data as data_mod
import models as model_mod
import metrics as metric_mod

warnings.filterwarnings("ignore")

y_te_cache = {"y": None}


def _fit_predict(model, X_tr, y_tr, X_te, task):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    if task == "fault":
        proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
        return metric_mod.classification_metrics(y_te_cache["y"], y_pred, proba)
    return metric_mod.regression_metrics(y_te_cache["y"], y_pred)


def _score(model, X_tr, y_tr, X_te, y_te, task):
    y_te_cache["y"] = y_te
    return _fit_predict(model, X_tr, y_tr, X_te, task)


def run_o1(df, task, use_weather):
    zoo = model_mod.get_regressors() if task == "power" else model_mod.get_classifiers()
    rows = []
    for site in config.ALL_SITES:
        X_tr, X_te, y_tr, y_te = data_mod.within_site_split(df, site, task, use_weather)
        for name, model in zoo.items():
            m = _score(model, X_tr, y_tr, X_te, y_te, task)
            rows.append({"experiment": "O1_within", "task": task, "site": site, "model": name, **m})
    return pd.DataFrame(rows)


def run_o2(df, task, use_weather):
    zoo = model_mod.get_regressors() if task == "power" else model_mod.get_classifiers()
    rows = []
    for test_site in config.TEST_SITES:
        X_tr, X_te, y_tr, y_te = data_mod.cross_site_split(
            df, config.TRAIN_SITES, [test_site], task, use_weather)
        for name, model in zoo.items():
            m = _score(model, X_tr, y_tr, X_te, y_te, task)
            rows.append({"experiment": "O2_cross", "task": task,
                         "site": f"{'+'.join(config.TRAIN_SITES)}->{test_site}", "model": name, **m})
    return pd.DataFrame(rows)


def degradation_report(o1, o2, task):
    key = "R2" if task == "power" else "F1"
    train_site = config.TRAIN_SITES[0]
    base = o1[o1.site == train_site].set_index("model")[key]
    rows = []
    for _, r in o2.iterrows():
        within = base.get(r["model"], np.nan)
        rows.append({"task": task, "model": r["model"], "test": r["site"],
                     f"{key}_within": round(within, 4), f"{key}_cross": round(r[key], 4),
                     f"{key}_drop": round(within - r[key], 4)})
    return pd.DataFrame(rows)


def _print_df(title, df):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    with pd.option_context("display.width", 120, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="PV cross-site baseline benchmark")
    ap.add_argument("--task", choices=["power", "fault", "both"], default="both")
    ap.add_argument("--experiment", choices=["o1", "o2", "both"], default="both")
    ap.add_argument("--weather", action="store_true", help="include weather features (Angle B)")
    ap.add_argument("--train", default=None, help="comma-sep train sites (overrides config.TRAIN_SITES)")
    ap.add_argument("--test", default=None, help="comma-sep test sites (overrides config.TEST_SITES)")
    args = ap.parse_args()

    if args.train:
        config.TRAIN_SITES = args.train.split(",")
    if args.test:
        config.TEST_SITES = args.test.split(",")

    print(model_mod.availability_note())
    print(f"[benchmark] train={config.TRAIN_SITES} -> test={config.TEST_SITES}")
    df = data_mod.load_data()
    print(f"[data] loaded {len(df):,} rows across sites {sorted(df[config.SITE_COL].unique())}; "
          f"weather={'ON' if args.weather else 'OFF'}")

    tasks = ["power", "fault"] if args.task == "both" else [args.task]
    tag = "weather" if args.weather else "noweather"

    for task in tasks:
        o1 = o2 = None
        if args.experiment in ("o1", "both"):
            o1 = run_o1(df, task, args.weather)
            _print_df(f"O1 within-site  |  task={task}", o1)
            o1.to_csv(config.RESULTS_DIR / f"o1_{task}_{tag}.csv", index=False)
        if args.experiment in ("o2", "both"):
            o2 = run_o2(df, task, args.weather)
            _print_df(f"O2 cross-site (zero-shot)  |  task={task}", o2)
            o2.to_csv(config.RESULTS_DIR / f"o2_{task}_{tag}.csv", index=False)
        if o1 is not None and o2 is not None:
            deg = degradation_report(o1, o2, task)
            _print_df(f"DEGRADATION (within -> cross)  |  task={task}  [headline result]", deg)
            deg.to_csv(config.RESULTS_DIR / f"degradation_{task}_{tag}.csv", index=False)

    print(f"\n[done] CSVs written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
