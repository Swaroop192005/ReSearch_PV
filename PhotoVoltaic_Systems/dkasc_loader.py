"""
DKASC data loader  (Desert Knowledge Australia Solar Centre)
============================================================

Turns a folder of DKASC per-array CSV exports into ONE clean multi-site CSV in
the schema the rest of the pipeline expects (a `site` column + `power` target +
weather features).

Columns are AUTO-DETECTED by matching known name fragments (robust to DKASC's
spaces/underscores/units). After it runs it PRINTS the exact `config.py` lines
to paste, so you don't have to guess.

Handles DKASC's big files by RESAMPLING to hourly by default (millions of
per-minute rows -> a few thousand rows/site), which keeps training fast.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
1. Put the per-array CSVs in a folder, e.g. data/dkasc/ (one CSV per array).
   The site name comes from the file name (e.g. "Trina_monoSi.csv" -> "Trina_monoSi").

2. Run (hourly resample, optionally limit the years):
       python dkasc_loader.py --folder data/dkasc --out data/pv_multisite.csv
       python dkasc_loader.py --folder data/dkasc --resample 1h --start 2019-01-01 --end 2021-12-31

3. Paste the printed CONFIG SNIPPET into config.py, then:
       python benchmark.py --task power
       python transfer.py  --task power --target <site>
       python benchmark.py --task power --weather

Notes
-----
* DKASC arrays feed the POWER-PREDICTION (regression) task; they have no fault
  labels. Fault detection uses GPVS-Faults separately.
* `current` is intentionally NOT used as a feature (it mirrors power, which would
  make the task trivial). Features are weather + time -> power.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Canonical name -> candidate header fragments (matched after removing non-alnum).
COLUMN_CANDIDATES = {
    "timestamp":          ["timestamp", "time", "date"],
    "power":              ["activepower", "acpower", "power"],
    # base physical driver (always a feature)
    "irradiance":         ["globalhorizontalradiation", "globalhorizontal", "ghi",
                           "planeofarray", "poa", "globaltiltedradiation", "radiation", "irradiance"],
    # weather features (the O4 add-on)
    "ambient_temp":       ["weathertemperaturecelsius", "ambienttemperature",
                           "airtemperature", "temperaturecelsius", "temperature"],
    "humidity":           ["relativehumidity", "humidity"],
    "wind_speed":         ["windspeed"],
    "diffuse_irradiance": ["diffusehorizontalradiation", "diffuse"],
    "module_temp":        ["moduletemperature", "celltemperature", "paneltemperature"],
}
# Feature roles for the printed config snippet
BASE_FEATURES_PREF    = ["irradiance", "module_temp", "hour", "month"]
WEATHER_FEATURES_PREF = ["ambient_temp", "humidity", "wind_speed", "diffuse_irradiance"]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _match_columns(df):
    norm_map = {_norm(c): c for c in df.columns}
    found = {}
    for canon, subs in COLUMN_CANDIDATES.items():
        for sub in subs:
            hit = next((orig for n, orig in norm_map.items() if sub in n), None)
            if hit is not None:
                found[canon] = hit
                break
    return found


def load_one(path, site_name, resample="1h", start=None, end=None, normalize=True):
    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    m = _match_columns(df)
    if "power" not in m:
        print(f"  [skip] {os.path.basename(path)}: no power column detected. "
              f"Columns seen: {list(df.columns)[:8]} ...")
        return None, m

    frame = pd.DataFrame(index=df.index)
    for canon, col in m.items():
        if canon == "timestamp":
            continue
        frame[canon] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in m:
        # utc=True handles both tz-naive (DKASC) and tz-aware / DST (NIST) inputs;
        # tz_localize(None) then drops the tz so date filtering works uniformly.
        ts = pd.to_datetime(df[m["timestamp"]], errors="coerce", utc=True).dt.tz_localize(None)
        frame["_ts"] = ts
        frame = frame.dropna(subset=["_ts"])
        if start:
            frame = frame[frame["_ts"] >= pd.Timestamp(start)]
        if end:
            frame = frame[frame["_ts"] <= pd.Timestamp(end)]
        if resample:
            frame = frame.set_index("_ts").resample(resample).mean()
            frame["hour"] = frame.index.hour
            frame["month"] = frame.index.month
            frame = frame.reset_index(drop=True)
        else:
            frame["hour"] = frame["_ts"].dt.hour
            frame["month"] = frame["_ts"].dt.month
            frame = frame.drop(columns=["_ts"])
    else:
        frame["hour"] = 0
        frame["month"] = 0

    # kW -> W if it looks like kW
    if frame["power"].notna().any() and frame["power"].max() < 2000:
        frame["power"] = frame["power"] * 1000.0

    frame = frame[frame["power"].notna() & (frame["power"] > 0)]
    if "irradiance" in frame.columns:
        frame = frame[frame["irradiance"].fillna(0) > 20]  # daylight only
    frame = frame.dropna(axis=1, how="all")
    frame = frame.fillna(frame.median(numeric_only=True))

    # Normalize power to capacity factor (power / rated capacity, estimated by the
    # 99th percentile). This makes arrays of very different sizes -- e.g. a 5 kW
    # Alice Springs rooftop vs a 227 kW Yulara plant -- directly comparable, which
    # is essential for the cross-LOCATION experiment.
    if normalize:
        cap = frame["power"].quantile(0.99)
        if cap and cap > 0:
            frame["power"] = (frame["power"] / cap).clip(0, 1.2)

    frame.insert(0, "site", site_name)
    return frame.reset_index(drop=True), m


def build(folder, out_csv, resample, start, end, normalize=True, site_map=None):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {folder}")
    print(f"[loader] power normalization: {'ON (capacity factor 0-1)' if normalize else 'OFF (raw)'}")
    frames = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        site = (site_map or {}).get(base, base)
        print(f"- {os.path.basename(f)}  ->  site '{site}'")
        df, m = load_one(f, site, resample, start, end, normalize)
        if df is not None:
            print(f"    detected: {sorted(m.keys())}  ({len(df):,} rows after resample)")
            frames.append(df)
    if not frames:
        sys.exit("No usable files (no power column found in any).")

    combined = pd.concat(frames, ignore_index=True)
    # keep only feature columns present for ALL sites (fair cross-site comparison)
    common = set(combined.columns)
    for _, sub in combined.groupby("site"):
        common &= set(sub.dropna(axis=1, how="all").columns)
    combined = combined[[c for c in combined.columns if c in common]]
    combined.to_csv(out_csv, index=False)

    sites = sorted(combined["site"].unique())
    feats = [c for c in combined.columns if c not in ("site", "power")]
    base = [c for c in BASE_FEATURES_PREF if c in feats]
    weather = [c for c in WEATHER_FEATURES_PREF if c in feats]
    base += [c for c in feats if c not in base + weather]

    print("\n" + "=" * 70)
    print(f"Wrote {len(combined):,} rows to {out_csv}")
    print(f"Sites: {sites}")
    print(f"Rows/site: {combined.groupby('site').size().to_dict()}")
    print("=" * 70)
    print("\n----- CONFIG SNIPPET (paste into config.py) -----\n")
    print('SITE_COL  = "site"')
    print('POWER_COL = "power"')
    print('FAULT_COL = "fault"   # (unused for DKASC power task)')
    print(f"ELECTRICAL_FEATURES = {base}")
    print(f"WEATHER_FEATURES    = {weather}")
    print(f"ALL_SITES   = {sites}")
    print(f'TRAIN_SITES = ["{sites[0]}"]')
    print(f"TEST_SITES  = {sites[1:] if len(sites) > 1 else sites}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a multi-site CSV from DKASC exports")
    ap.add_argument("--folder", required=True, help="folder of DKASC per-array CSVs")
    ap.add_argument("--out", default="data/pv_multisite.csv", help="output CSV path")
    ap.add_argument("--resample", default="1h", help="pandas resample freq (e.g. 1h, 30min, D). '' disables.")
    ap.add_argument("--start", default=None, help="start date YYYY-MM-DD (optional)")
    ap.add_argument("--end", default=None, help="end date YYYY-MM-DD (optional)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="keep raw power in W instead of capacity factor (not recommended for mixed-size arrays)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    build(args.folder, args.out, args.resample or None, args.start, args.end,
          normalize=not args.no_normalize)
