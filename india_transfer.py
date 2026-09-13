"""
Five-method adaptation comparison with the Indian plant as target -- the exact
analogue of the paper's Table IV (transfer_advanced.py). With seed=42 the split
and few-shot draw are identical to transfer_advanced.py's single run; the other
four seeds show the spread on a small (412 h) target.

Sources: Trina (the paper's Table IV source), NIST, and all four AU/US arrays pooled.

Run:  python india_transfer.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from india_common import (AUUS, INDIA, RESULTS, SEEDS, catboost, load_all,
                          paper_mlp_finetune, paper_mlp_pretrained, xy)

warnings.filterwarnings("ignore")


# --- verbatim from transfer_advanced.py ---
def _matpow(M, p, eps=1e-6):
    M = M + eps * np.eye(M.shape[0]); w, V = np.linalg.eigh(M); w = np.clip(w, 1e-10, None)
    return (V * (w ** p)) @ V.T


def coral(Xs, Xt):
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-8
    Xs_z, Xt_z = (Xs - mu) / sd, (Xt - mu) / sd
    if Xs.shape[1] < 2:
        return Xs_z, Xt_z
    A = _matpow(np.cov(Xs_z, rowvar=False), -0.5) @ _matpow(np.cov(Xt_z, rowvar=False), 0.5)
    return Xs_z @ A, Xt_z


def imp_weights(Xs, Xt):
    Xd = np.vstack([Xs, Xt]); yd = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = HistGradientBoostingClassifier(random_state=0).fit(Xd, yd)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-3, 1 - 1e-3)
    w = p / (1 - p); return w / w.mean()
# ------------------------------------------


df = load_all()
SOURCES = {"Trina_monoSi": ["Trina_monoSi"],
           "NIST_Ground_monoSi": ["NIST_Ground_monoSi"],
           "AU+US pooled": AUUS}
rows = []
for weather in (False, True):
    wtag = "w" if weather else "nw"
    Xt, yt = xy(df, [INDIA], weather)
    for sname, slist in SOURCES.items():
        Xs, ys = xy(df, slist, weather)
        zero_model = catboost(42).fit(Xs, ys)
        sc, net = paper_mlp_pretrained(Xs, ys)
        for seed in SEEDS:
            Xpool, Xte, ypool, yte = train_test_split(Xt, yt, test_size=0.4, random_state=seed)
            zero = r2_score(yte, zero_model.predict(Xte))
            within = r2_score(yte, catboost(42).fit(Xpool, ypool).predict(Xte))
            Xs_al, Xte_z = coral(Xs.values, Xte.values)
            cor = r2_score(yte, catboost(42).fit(Xs_al, ys).predict(Xte_z))
            w = imp_weights(Xs.values, Xpool.values)
            iw = r2_score(yte, catboost(42).fit(Xs, ys, sample_weight=w).predict(Xte))
            rng = np.random.default_rng(seed)
            k = min(max(5, int(0.2 * len(Xt))), len(Xpool))
            idx = rng.choice(len(Xpool), size=k, replace=False)
            Xk, yk = Xpool.iloc[idx], ypool[idx]
            fs = r2_score(yte, catboost(42).fit(pd.concat([Xs, Xk], ignore_index=True),
                                                 np.concatenate([ys, yk])).predict(Xte))
            mlp = r2_score(yte, paper_mlp_finetune(sc, net, Xk, yk).predict(sc.transform(Xte)))
            gap = within - zero
            for meth, v in (("zero_shot", zero), ("CORAL", cor), ("importance_weight", iw),
                            ("few_shot_tree", fs), ("mlp_finetune", mlp), ("within_target", within)):
                rows.append(dict(weather=wtag, source=sname, seed=seed, k=k, method=meth,
                                 R2=v, gap=gap, recovery=(v - zero) / gap))
        print(f"[india_transfer] {wtag} {sname} done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "india_transfer_raw.csv", index=False)
order = ["zero_shot", "CORAL", "importance_weight", "few_shot_tree", "mlp_finetune", "within_target"]
for (w, s), g in R.groupby(["weather", "source"], sort=False):
    print(f"\n===== {s} -> India | weather={'ON' if w == 'w' else 'OFF'} | 20% target (k={g.k.iloc[0]}) =====")
    print(f"  {'method':20s}{'seed42 R2':>11s}{'seed42 rec':>12s}{'mean R2':>10s}{'mean rec':>10s}{'rec sd':>8s}")
    for meth in order:
        gm = g[g.method == meth]; g42 = gm[gm.seed == 42].iloc[0]
        print(f"  {meth:20s}{g42.R2:>11.4f}{100 * g42.recovery:>11.1f}%"
              f"{gm.R2.mean():>10.4f}{100 * gm.recovery.mean():>9.1f}%{100 * gm.recovery.std():>7.1f}")
    print(f"  gap (ceiling - zero-shot): seed42 {g[g.seed == 42].gap.iloc[0]:.4f}, mean {g.drop_duplicates('seed').gap.mean():.4f}")
