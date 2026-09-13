"""
How much local data closes the AU/US -> India gap? Sweeps k target samples
(nested subsets of one per-seed permutation) and compares three repairs, all
measured against the paper's CatBoost zero-shot and CatBoost ceiling:
  * gain   : one least-squares multiplicative factor on the zero-shot predictions
  * tree   : few-shot CatBoost (source + k target rows), as in the paper
  * mlp    : the paper's MLP fine-tune on the k rows

Run:  python india_recovery.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from india_common import (AUUS, INDIA, RESULTS, SEEDS, catboost, load_all,
                          paper_mlp_finetune, paper_mlp_pretrained, xy)

warnings.filterwarnings("ignore")
KS = [5, 10, 20, 40, 82, 165, 247]
SOURCES = {"Trina_monoSi": ["Trina_monoSi"], "AU+US pooled": AUUS}

df = load_all()
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
            pz_pool, pz_te = zero_model.predict(Xpool), zero_model.predict(Xte)
            zero = r2_score(yte, pz_te)
            within = r2_score(yte, catboost(42).fit(Xpool, ypool).predict(Xte))
            gap = within - zero
            perm = np.random.default_rng(seed).permutation(len(Xpool))
            for k in KS:
                idx = perm[:min(k, len(Xpool))]
                a = float(np.sum(pz_pool[idx] * ypool[idx]) / np.sum(pz_pool[idx] ** 2))
                gain = r2_score(yte, a * pz_te)
                tree = r2_score(yte, catboost(42).fit(
                    pd.concat([Xs, Xpool.iloc[idx]], ignore_index=True),
                    np.concatenate([ys, ypool[idx]])).predict(Xte))
                mlp = r2_score(yte, paper_mlp_finetune(sc, net, Xpool.iloc[idx], ypool[idx])
                               .predict(sc.transform(Xte)))
                for meth, v in (("gain", gain), ("tree", tree), ("mlp", mlp)):
                    rows.append(dict(weather=wtag, source=sname, seed=seed, k=k, method=meth,
                                     R2=v, zero=zero, within=within, gap=gap,
                                     recovery=(v - zero) / gap, gain_a=a))
        print(f"[india_recovery] {wtag} {sname} done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "india_recovery_raw.csv", index=False)
for (w, s), g in R.groupby(["weather", "source"], sort=False):
    z = g.drop_duplicates("seed")
    print(f"\n===== {s} -> India | weather={'ON' if w == 'w' else 'OFF'} | zero-shot {z.zero.mean():.4f} "
          f"ceiling {z.within.mean():.4f} gap {z.gap.mean():.4f} (5 seeds) =====")
    print(f"  {'k':>4s}{'gain R2':>9s}{'rec':>7s}{'tree R2':>9s}{'rec':>7s}{'mlp R2':>9s}{'rec':>7s}{'a':>8s}")
    for k in KS:
        gk = g[g.k == k]
        m = {meth: gk[gk.method == meth] for meth in ("gain", "tree", "mlp")}
        print(f"  {k:>4d}" + "".join(f"{m[x].R2.mean():>9.4f}{100 * m[x].recovery.mean():>6.0f}%"
                                      for x in ("gain", "tree", "mlp"))
              + f"{m['gain'].gain_a.mean():>8.3f}")
