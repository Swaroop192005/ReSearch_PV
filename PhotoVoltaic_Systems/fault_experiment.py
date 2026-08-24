"""
fault_experiment.py -- within-mode vs cross-mode fault detection on GPVS-Faults.

Binary fault detection (healthy vs faulty). Trains under one operating mode and
tests under the same (within) or the other (cross: MPPT<->IPPT), over several
seeds. Reports accuracy/precision/recall/F1/AUC and the cross-mode gap, with a
significance test -- the fault-detection analog of the power generalization study.

Usage:
  python fault_experiment.py --features data/gpvs_features.csv
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from scipy import stats
import config, models as m, metrics as met
warnings.filterwarnings("ignore")
SEEDS = [42,1,2,7,123]

def pick():
    zoo = m.get_classifiers()
    for n in ["CatBoost","XGBoost","HistGradientBoosting","GradientBoosting","RandomForest"]:
        if n in zoo: return n, zoo[n]
    n = next(iter(zoo)); return n, zoo[n]

def score(model, Xtr, ytr, Xte, yte, seed):
    mm = clone(model)
    try: mm.set_params(random_state=seed)
    except Exception: pass
    mm.fit(Xtr, ytr); yp = mm.predict(Xte)
    pr = mm.predict_proba(Xte)[:,1] if hasattr(mm,"predict_proba") else None
    return met.classification_metrics(yte, yp, pr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/gpvs_features.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.features)
    feat = [c for c in df.columns if c not in ("fault","scenario","mode")]
    name, base = pick()
    modes = ["MPPT","IPPT"]
    print(f"[fault] model={name}  windows={len(df)}  features={len(feat)}")
    print(df.groupby(["mode","fault"]).size(), "\n")
    rec = []
    for seed in SEEDS:
        for src in modes:
            for tgt in modes:
                if src == tgt:
                    X = df[df["mode"]==src][feat]; y = df[df["mode"]==src]["fault"]
                    xa,xb,ya,yb = train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
                    m_ = score(base, xa, ya, xb, yb, seed); typ="within"
                else:
                    Xtr = df[df["mode"]==src][feat]; ytr = df[df["mode"]==src]["fault"]
                    Xte = df[df["mode"]==tgt][feat]; yte = df[df["mode"]==tgt]["fault"]
                    m_ = score(base, Xtr, ytr, Xte, yte, seed); typ="cross"
                rec.append({"seed":seed,"src":src,"tgt":tgt,"type":typ,**m_})
    R = pd.DataFrame(rec)
    print("===== FAULT DETECTION: mean metrics by setting =====")
    g = R.groupby(["type","src","tgt"])[["Accuracy","F1","AUC","MCC","Kappa"]].mean().round(4)
    print(g)
    win = R[R.type=="within"]["F1"].values; cro = R[R.type=="cross"]["F1"].values
    print(f"\nwithin-mode  F1 = {win.mean():.4f} +/- {win.std(ddof=1):.4f}")
    print(f"cross-mode   F1 = {cro.mean():.4f} +/- {cro.std(ddof=1):.4f}")
    print(f"cross-mode gap  = {win.mean()-cro.mean():.4f}")
    U,p = stats.mannwhitneyu(win, cro, alternative="greater")
    print(f"significance (within>cross): Mann-Whitney p = {p:.2e}")
    R.to_csv(config.RESULTS_DIR/"fault_results.csv", index=False)
    print(f"[done] -> {config.RESULTS_DIR}/fault_results.csv")

if __name__ == "__main__": main()
