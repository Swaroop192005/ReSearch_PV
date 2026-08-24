"""
transfer_advanced.py -- compare domain-adaptation methods for cross-site transfer.

For a source -> target pair, on a fixed target-test split, it evaluates:
  * zero_shot            : source-only model (lower bound)
  * CORAL                : unsupervised 2nd-order feature alignment (no target labels)
  * importance_weighting : reweight source samples by a domain-classifier density
                           ratio, then train weighted source model (no target labels)
  * few_shot_tree        : pool source + k% target labels, retrain the tree model
  * mlp_finetune         : train an MLP on source, then fine-tune (warm-start) on k%
                           target labels -- a neural transfer baseline
  * within_target        : model trained on the full target pool (upper bound)

Recovery = (method - zero_shot) / (within_target - zero_shot).

Usage:
  python transfer_advanced.py --task power --source Trina_monoSi --target NIST_Ground_monoSi --frac 0.2
  python transfer_advanced.py --task power --source Trina_monoSi --target NIST_Ground_monoSi --frac 0.2 --weather
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import config, data as d, models as m, metrics as met
warnings.filterwarnings("ignore")
KEY = {"power": "R2", "fault": "F1"}


def base_model(task):
    zoo = m.get_regressors() if task == "power" else m.get_classifiers()
    for n in ["CatBoost","XGBoost","HistGradientBoosting","GradientBoosting","RandomForest"]:
        if n in zoo: return n, zoo[n]
    n = next(iter(zoo)); return n, zoo[n]


def evalr(model, Xtr, ytr, Xte, yte, task, w=None):
    mm = clone(model)
    try:
        mm.fit(Xtr, ytr, sample_weight=w) if w is not None else mm.fit(Xtr, ytr)
    except TypeError:
        mm.fit(Xtr, ytr)
    yp = mm.predict(Xte)
    if task == "fault":
        pr = mm.predict_proba(Xte)[:, 1] if hasattr(mm, "predict_proba") else None
        return met.classification_metrics(yte, yp, pr)[KEY[task]]
    return met.regression_metrics(yte, yp)[KEY[task]]


def _matpow(M, p, eps=1e-6):
    M = M + eps*np.eye(M.shape[0]); w, V = np.linalg.eigh(M); w = np.clip(w, 1e-10, None)
    return (V*(w**p)) @ V.T

def coral(Xs, Xt):
    mu, sd = Xs.mean(0), Xs.std(0)+1e-8
    Xs_z, Xt_z = (Xs-mu)/sd, (Xt-mu)/sd
    if Xs.shape[1] < 2: return Xs_z, Xt_z
    A = _matpow(np.cov(Xs_z, rowvar=False), -0.5) @ _matpow(np.cov(Xt_z, rowvar=False), 0.5)
    return Xs_z @ A, Xt_z

def imp_weights(Xs, Xt):
    Xd = np.vstack([Xs, Xt]); yd = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = HistGradientBoostingClassifier(random_state=0).fit(Xd, yd)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-3, 1-1e-3)
    w = p/(1-p); return w/w.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["power","fault"], default="power")
    ap.add_argument("--source", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--frac", type=float, default=0.2, help="target labels used by supervised methods")
    ap.add_argument("--weather", action="store_true")
    args = ap.parse_args()

    df = d.load_data()
    name, base = base_model(args.task)
    Xs, ys = d.get_xy(d.site_frame(df, [args.source]), args.task, args.weather)
    Xt, yt = d.get_xy(d.site_frame(df, [args.target]), args.task, args.weather)
    feats = d.feature_columns(args.weather)
    strat = yt if args.task == "fault" else None
    Xpool, Xte, ypool, yte = train_test_split(Xt, yt, test_size=0.4, random_state=config.RANDOM_STATE, stratify=strat)

    zero = evalr(base, Xs, ys, Xte, yte, args.task)
    within = evalr(base, Xpool, ypool, Xte, yte, args.task)
    gap = within - zero
    rows = [("zero_shot", zero)]

    # CORAL (unsupervised)
    Xs_al, Xte_z = coral(Xs.values, Xte.values)
    rows.append(("CORAL (unsup.)", evalr(base, pd.DataFrame(Xs_al,columns=feats), ys, pd.DataFrame(Xte_z,columns=feats), yte, args.task)))

    # Importance weighting (unsupervised)
    w = imp_weights(Xs.values, Xpool.values)
    rows.append(("importance_weight (unsup.)", evalr(base, Xs, ys, Xte, yte, args.task, w=w)))

    # Few-shot tree (supervised, k% target)
    rng = np.random.default_rng(config.RANDOM_STATE)
    k = max(5, int(args.frac*len(Xt))); k = min(k, len(Xpool))
    idx = rng.choice(len(Xpool), size=k, replace=False)
    Xk, yk = Xpool.iloc[idx], ypool[idx]
    Xcomb = pd.concat([Xs, Xk], ignore_index=True); ycomb = np.concatenate([ys, yk])
    rows.append((f"few_shot_tree ({int(args.frac*100)}%)", evalr(base, Xcomb, ycomb, Xte, yte, args.task)))

    # MLP fine-tune (supervised)
    sc = StandardScaler().fit(Xs)
    Net = MLPRegressor if args.task=="power" else MLPClassifier
    net = Net(hidden_layer_sizes=(32,16), max_iter=400, random_state=0, warm_start=True)
    net.fit(sc.transform(Xs), ys)
    net.max_iter = 200
    net.fit(sc.transform(Xk), yk)  # warm-start fine-tune on target subset
    yp = net.predict(sc.transform(Xte))
    if args.task=="power":
        mlp_score = met.regression_metrics(yte, yp)["R2"]
    else:
        pr = net.predict_proba(sc.transform(Xte))[:,1] if hasattr(net,"predict_proba") else None
        mlp_score = met.classification_metrics(yte, yp, pr)["F1"]
    rows.append((f"mlp_finetune ({int(args.frac*100)}%)", mlp_score))

    rows.append(("within_target (ceiling)", within))

    print(f"\n=== TL method comparison | {args.source} -> {args.target} | model={name} | "
          f"weather={'ON' if args.weather else 'OFF'} | target labels={int(args.frac*100)}% ===")
    print(f"{'method':30s} {KEY[args.task]:>8s} {'recovery':>10s}")
    out=[]
    for nm, sc_ in rows:
        rec = (sc_-zero)/gap if abs(gap)>1e-9 and nm not in("zero_shot",) else (1.0 if nm.startswith("within") else 0.0)
        print(f"{nm:30s} {sc_:8.4f} {rec*100:9.1f}%")
        out.append({"method":nm, KEY[args.task]:round(sc_,4), "recovery":round(rec,3)})
    pd.DataFrame(out).to_csv(config.RESULTS_DIR/f"tl_compare_{args.task}_{args.target}_{'w' if args.weather else 'nw'}.csv", index=False)
    print(f"[done] gap={gap:.4f}; written to results/")


if __name__ == "__main__":
    main()
