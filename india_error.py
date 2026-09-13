"""
Where does the AU/US -> India error sit? Exact analogue of error_analysis.py
(the paper's Table V / Fig. 6): CatBoost trained on the source and applied to
every Indian daylight hour; within-site predictions from 5-fold
cross_val_predict on the Indian plant; RMSE and bias by irradiance band, plus a
linear calibration fit.

Run:  python india_error.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from india_common import AUUS, INDIA, RESULTS, catboost, load_all, xy

warnings.filterwarnings("ignore")
BINS = [0, 200, 400, 600, 800, 1000, 3000]
LABELS = ["0-200", "200-400", "400-600", "600-800", "800-1000", ">1000"]

df = load_all()
rows = []
for weather in (False, True):
    wtag = "w" if weather else "nw"
    for sname, slist in (("Trina_monoSi", ["Trina_monoSi"]), ("AU+US pooled", AUUS)):
        Xs, ys = xy(df, slist, weather)
        Xt, yt = xy(df, [INDIA], weather)
        cross = catboost(42).fit(Xs, ys).predict(Xt)
        within = cross_val_predict(catboost(42), Xt, yt, cv=5)
        rc, rw = cross - yt, within - yt
        a, b0 = np.polyfit(cross, yt, 1)
        cat = pd.cut(Xt["irradiance"].values, bins=BINS, labels=LABELS, include_lowest=True)
        print(f"\n=== ERROR BY IRRADIANCE | {sname} -> India | weather={'ON' if weather else 'OFF'} | CatBoost ===")
        print(f"{'irr(W/m2)':11s}{'n':>6s}{'RMSE within':>13s}{'RMSE cross':>12s}{'bias cross':>12s}")
        for lab in LABELS:
            m = np.asarray(cat == lab)
            if m.sum() == 0:
                continue
            r = dict(weather=wtag, source=sname, band=lab, n=int(m.sum()),
                     rmse_within=float(np.sqrt(np.mean(rw[m] ** 2))),
                     rmse_cross=float(np.sqrt(np.mean(rc[m] ** 2))),
                     bias_cross=float(np.mean(rc[m])))
            rows.append(r)
            print(f"{lab:11s}{r['n']:>6d}{r['rmse_within']:>13.4f}{r['rmse_cross']:>12.4f}{r['bias_cross']:>+12.4f}")
        print(f"  overall RMSE within {np.sqrt(np.mean(rw ** 2)):.4f}  cross {np.sqrt(np.mean(rc ** 2)):.4f}")
        print(f"  calibration: actual = {a:.3f} * pred {b0:+.3f};  mean bias {rc.mean():+.4f} CF;  "
              f"under-predicts {100 * np.mean(rc < 0):.0f}% of hours")
        rows.append(dict(weather=wtag, source=sname, band="ALL", n=len(yt),
                         rmse_within=float(np.sqrt(np.mean(rw ** 2))),
                         rmse_cross=float(np.sqrt(np.mean(rc ** 2))),
                         bias_cross=float(rc.mean()), slope=a, intercept=b0,
                         under_pct=float(100 * np.mean(rc < 0))))

pd.DataFrame(rows).to_csv(RESULTS / "india_error_by_irradiance.csv", index=False)
