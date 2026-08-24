"""
gpvs_prep.py -- turn GPVS-Faults time-series files into a windowed feature table.

Each file Fxy.csv (x=0..7 fault scenario, y=L/M operating mode) is sliced into
non-overlapping windows; per window we compute mean/std/RMS of each electrical
signal. Labels: fault=0 for F0 (healthy), 1 for F1-F7; plus scenario and mode.
To avoid the pre-fault period, F0 uses the whole record (minus startup) while
F1-F7 use the LATTER HALF (fault active).

Usage:
  python gpvs_prep.py --folder ~/Downloads/gpvs/csv/CSV_Files --out data/gpvs_features.csv
"""
import argparse, glob, os, re
import numpy as np, pandas as pd

SIGNALS = ["Ipv","Vpv","Vdc","ia","ib","ic","va","vb","vc","Iabc","If","Vabc","Vf"]

def window_feats(win):
    out = {}
    for s in SIGNALS:
        if s not in win.columns: continue
        x = pd.to_numeric(win[s], errors="coerce").values.astype(float)
        x = x[~np.isnan(x)]
        if len(x) == 0: x = np.array([0.0])
        out[f"{s}_mean"] = x.mean(); out[f"{s}_std"] = x.std(); out[f"{s}_rms"] = np.sqrt((x*x).mean())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="data/gpvs_features.csv")
    ap.add_argument("--window", type=int, default=1000)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.folder, "F*.csv")))
    if not files: raise SystemExit(f"No Fxy.csv files in {args.folder}")
    rows = []
    for f in files:
        base = os.path.basename(f); mo = re.match(r"F(\d)([LM])", base)
        if not mo: continue
        scen = int(mo.group(1)); mode = "MPPT" if mo.group(2) == "M" else "IPPT"
        df = pd.read_csv(f); n = len(df)
        lo = int(0.10*n) if scen == 0 else int(0.50*n)   # skip startup / pre-fault
        df = df.iloc[lo:].reset_index(drop=True); W = args.window
        cnt = 0
        for i in range(0, len(df)-W+1, W):
            r = window_feats(df.iloc[i:i+W])
            r["fault"] = 0 if scen == 0 else 1; r["scenario"] = scen; r["mode"] = mode
            rows.append(r); cnt += 1
        print(f"  {base}: scenario={scen} mode={mode} windows={cnt}")
    out = pd.DataFrame(rows).dropna(axis=1, how="all")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[gpvs] wrote {len(out)} windows x {out.shape[1]-3} features -> {args.out}")
    print(out.groupby(["mode","fault"]).size())

if __name__ == "__main__": main()
