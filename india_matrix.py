"""
All-pairs extension to India: every AU/US array -> the Indian plant, and back.
Identical protocol to matrix_experiment.py (CatBoost, 5 seeds, 80/20 within-site
split, cross-site scored on the whole target). AU<->US cells come from the
paper's own reproduced run (results/matrix_raw_power_{nw,w}.csv).

Run:  python india_matrix.py
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from india_common import AUUS, INDIA, RESULTS, SEEDS, catboost, load_all, xy

warnings.filterwarnings("ignore")


def ci95(x):
    x = np.asarray(x, float)
    h = x.std(ddof=1) / np.sqrt(len(x)) * stats.t.ppf(0.975, len(x) - 1)
    return x.mean() - h, x.mean() + h


df = load_all()
sites = AUUS + [INDIA]
rows = []
for weather in (False, True):
    wtag = "w" if weather else "nw"
    for seed in SEEDS:
        within = {}
        for s in sites:
            X, y = xy(df, [s], weather)
            xa, xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed)
            within[s] = r2_score(yb, catboost(seed).fit(xa, ya).predict(xb))
        rows.append(dict(weather=wtag, seed=seed, source=INDIA, target=INDIA,
                         direction="within", R2=within[INDIA],
                         within_target=within[INDIA], gap=0.0))
        for src, tgt in itertools.permutations(sites, 2):
            if INDIA not in (src, tgt):
                continue
            Xs, ys = xy(df, [src], weather)
            Xt, yt = xy(df, [tgt], weather)
            r = r2_score(yt, catboost(seed).fit(Xs, ys).predict(Xt))
            rows.append(dict(weather=wtag, seed=seed, source=src, target=tgt,
                             direction="to-India" if tgt == INDIA else "from-India",
                             R2=r, within_target=within[tgt], gap=within[tgt] - r))
    print(f"[india_matrix] weather={wtag} done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "india_matrix_raw.csv", index=False)

for w in ("nw", "w"):
    paper = pd.read_csv(RESULTS / f"matrix_raw_power_{w}.csv")
    clim = paper[paper["type"] == "cross-climate"]["gap"].values
    tech = paper[paper["type"] == "cross-technology"]["gap"].values
    sub = R[R["weather"] == w]
    print(f"\n===== INDIA ALL-PAIRS | weather={'ON' if w == 'w' else 'OFF'} | CatBoost, 5 seeds =====")
    print(f"  India within-site R2 (80/20): {sub[sub.direction == 'within'].R2.mean():.4f}")
    print(f"  paper AU<->US cross-climate gap {clim.mean():.4f} (n={len(clim)}), "
          f"cross-technology {tech.mean():.4f}")
    for dirn, key in (("to-India", "source"), ("from-India", "target")):
        g = sub[sub["direction"] == dirn]
        lo, hi = ci95(g["gap"])
        print(f"  {dirn:10s} gap: n={len(g)} mean={g.gap.mean():.4f} "
              f"95%CI=[{lo:.4f}, {hi:.4f}]  = {g.gap.mean() / clim.mean():.2f}x AU<->US")
        for s, gg in g.groupby(key):
            print(f"      {s:20s} R2={gg.R2.mean():.4f}  gap={gg.gap.mean():.4f}")
    g_in = sub[sub["direction"] == "to-India"]["gap"].values
    U, p = stats.mannwhitneyu(g_in, clim, alternative="greater")
    t, pt = stats.ttest_ind(g_in, clim, equal_var=False)
    print(f"  to-India gap > AU<->US gap?  Mann-Whitney U={U:.0f} p={p:.2e}; "
          f"Welch t={t:.2f} p={pt:.2e}")
