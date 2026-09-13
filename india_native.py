"""
(1) The India-native prior: the paper's CatBoost and MLP trained only on 500,403
    PVGIS hours across 10 Indian climate zones, against the AU/US-measured prior,
    both scored on the same held-out Kaggle Plant 1 split (5 seeds).
(2) Leave-one-zone-out across the 10 zones with the paper's CatBoost.

Run:  python india_native.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from india_common import (AUUS, INDIA, PVGIS_CSV, RESULTS, SEEDS, catboost, feats,
                          load_auus, load_india, paper_mlp_finetune,
                          paper_mlp_pretrained, xy)

warnings.filterwarnings("ignore")
pv = pd.read_csv(PVGIS_CSV)
auus, india = load_auus(), load_india()

# ---------------- (1) head-to-head ----------------
rows = []
for weather in (False, True):
    wtag = "w" if weather else "nw"
    Xpv = pv[feats(weather)].astype(float).reset_index(drop=True)
    ypv = pv["power"].astype(float).values
    Xau, yau = xy(auus, AUUS, weather)
    Xt, yt = xy(india, [INDIA], weather)
    priors = {}
    for name, (Xs, ys) in (("AU+US measured", (Xau, yau)), ("India-native PVGIS", (Xpv, ypv))):
        cb = catboost(42).fit(Xs, ys)
        sc, net = paper_mlp_pretrained(Xs, ys)
        priors[name] = (Xs, ys, cb, sc, net)
        print(f"[native] {wtag} trained {name} on {len(ys):,} rows", flush=True)
    for seed in SEEDS:
        Xpool, Xte, ypool, yte = train_test_split(Xt, yt, test_size=0.4, random_state=seed)
        within = r2_score(yte, catboost(42).fit(Xpool, ypool).predict(Xte))
        rng = np.random.default_rng(seed)
        k = min(max(5, int(0.2 * len(Xt))), len(Xpool))
        idx = rng.choice(len(Xpool), size=k, replace=False)
        Xk, yk = Xpool.iloc[idx], ypool[idx]
        for name, (Xs, ys, cb, sc, net) in priors.items():
            rec = dict(weather=wtag, prior=name, seed=seed, within=within,
                       zs_catboost=r2_score(yte, cb.predict(Xte)),
                       zs_mlp=r2_score(yte, net.predict(sc.transform(Xte))),
                       mlp_ft20=r2_score(yte, paper_mlp_finetune(sc, net, Xk, yk)
                                         .predict(sc.transform(Xte))),
                       tree_fs20=r2_score(yte, catboost(42).fit(
                           pd.concat([Xs, Xk], ignore_index=True),
                           np.concatenate([ys, yk])).predict(Xte)))
            rows.append(rec)
        print(f"[native] {wtag} seed {seed} done", flush=True)
H = pd.DataFrame(rows)
H.to_csv(RESULTS / "india_native_raw.csv", index=False)
print("\n===== PRIOR HEAD-TO-HEAD on held-out Kaggle Plant 1 (mean of 5 seeds) =====")
print(H.groupby(["weather", "prior"])[["zs_catboost", "zs_mlp", "tree_fs20", "mlp_ft20", "within"]]
      .mean().round(4).to_string())

# ---------------- (2) leave-one-zone-out ----------------
lz = []
for weather in (False, True):
    wtag = "w" if weather else "nw"
    f = feats(weather)
    for z in sorted(pv["zone"].unique()):
        tr, te = pv[pv["zone"] != z], pv[pv["zone"] == z]
        m = catboost(42).fit(tr[f].astype(float), tr["power"].values)
        r = r2_score(te["power"].values, m.predict(te[f].astype(float)))
        lz.append(dict(weather=wtag, zone=z, n_test=len(te), R2=r))
        print(f"[lozo] {wtag} {z:24s} R2={r:.4f}", flush=True)
L = pd.DataFrame(lz)
L.to_csv(RESULTS / "india_lozo_catboost.csv", index=False)
print("\n===== LEAVE-ONE-ZONE-OUT (CatBoost) =====")
print(L.pivot(index="zone", columns="weather", values="R2").round(4).to_string())
