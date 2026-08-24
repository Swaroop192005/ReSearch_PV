"""
temporal_experiment.py -- generalization over TIME (concept drift / aging).

For each DKASC array, train on one year and test on later years of the SAME
array, measuring how a model decays as panels age and conditions drift. Power is
normalized to capacity factor using a FIXED (train-year) capacity, so genuine
year-over-year decline in the irradiance->power relationship shows up as error.

Needs no new download: the DKASC per-array files already span all years.

Usage:
  python temporal_experiment.py --folder data/dkasc \
      --sites Trina_monoSi,Kyocera_polySi,Calyxo_CdTe \
      --train-year 2017 --test-years 2019,2021
"""
import argparse, glob, os, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import config, models as m, metrics as met
import dkasc_loader as dl
warnings.filterwarnings("ignore")

def pick():
    zoo = m.get_regressors()
    for n in ["CatBoost","XGBoost","HistGradientBoosting","GradientBoosting","RandomForest"]:
        if n in zoo: return n, zoo[n]
    n = next(iter(zoo)); return n, zoo[n]

def load_year(path, site, year):
    fy, _ = dl.load_one(path, site, "1h", f"{year}-01-01", f"{year}-12-31", normalize=False)
    return fy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/dkasc")
    ap.add_argument("--sites", default="Trina_monoSi,Kyocera_polySi,Calyxo_CdTe")
    ap.add_argument("--train-year", type=int, default=2017)
    ap.add_argument("--test-years", default="2019,2021")
    ap.add_argument("--weather", action="store_true")
    args = ap.parse_args()
    sites = args.sites.split(","); test_years = [int(y) for y in args.test_years.split(",")]
    years = [args.train_year] + [y for y in test_years if y != args.train_year]
    feats = ["irradiance"] + (["ambient_temp"] if args.weather else [])
    name, base = pick()
    print(f"[temporal] model={name}  train={args.train_year}  test={test_years}  weather={'ON' if args.weather else 'OFF'}")

    frames = []
    for site in sites:
        cand = glob.glob(os.path.join(args.folder, f"{site}*.csv"))
        if not cand: print(f"  [skip] no file for {site}"); continue
        path = cand[0]
        f0 = load_year(path, site, args.train_year)
        if f0 is None or len(f0) == 0: print(f"  [skip] {site}: no {args.train_year} data"); continue
        cap = f0["power"].quantile(0.99) or 1.0
        for yr in years:
            fy = load_year(path, site, yr)
            if fy is None or len(fy) == 0: print(f"  {site} {yr}: no data"); continue
            fy = fy.copy(); fy["power"] = fy["power"]/cap; fy["year"] = yr; fy["site"] = site
            frames.append(fy[["site","year","power"]+[c for c in feats if c in fy.columns]])
    if not frames: raise SystemExit("No data loaded -- check --folder and site names.")
    df = pd.concat(frames, ignore_index=True)
    print("\nrows per site/year:"); print(df.groupby(["site","year"]).size())

    rows = []
    for site in sites:
        s = df[df.site == site]
        tr = s[s.year == args.train_year]
        if len(tr) == 0: continue
        Xtr, ytr = tr[feats], tr["power"]
        xa, xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)
        within = met.regression_metrics(yb, clone(base).fit(xa, ya).predict(xb))["R2"]
        model = clone(base).fit(Xtr, ytr)
        for yr in years:
            sy = s[s.year == yr]
            if len(sy) == 0: continue
            r = met.regression_metrics(sy["power"], model.predict(sy[feats]))["R2"]
            rows.append({"site":site,"train":args.train_year,"test":yr,"gap_years":yr-args.train_year,
                         "R2":round(r,4),"drop_vs_within":round(within-r,4)})
    R = pd.DataFrame(rows)
    print(f"\n=== TEMPORAL GENERALIZATION (train {args.train_year}, fixed-capacity CF) ===")
    print(R.to_string(index=False))
    print("\nmean over sites, by gap (years):")
    print(R.groupby("gap_years")[["R2","drop_vs_within"]].mean().round(4))
    R.to_csv(config.RESULTS_DIR/"temporal_results.csv", index=False)
    print(f"[done] -> {config.RESULTS_DIR}/temporal_results.csv")

if __name__ == "__main__": main()
