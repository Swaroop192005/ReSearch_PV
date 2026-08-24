"""
nist_prep.py -- convert a NIST PV bulk download (daily CSVs under year/month/)
into ONE CSV in the same format as the DKASC files, so the normal loader can
ingest it alongside the Alice Springs arrays.

Picks the cleanest available power + irradiance columns:
  power       <- InvPDC_kW_Avg  (DC power; falls back to meter/AC power)
  irradiance  <- RefCell1_Wm2_Avg (plane-of-array; falls back to SEWS/Pyra)
  temperature <- AmbTemp_C_Avg
  wind        <- WindSpeedAve_ms

Usage:
  python nist_prep.py --folder ~/Downloads/nist_raw --out data/dkasc/NIST_Ground_monoSi.csv
"""
import argparse, glob, os, sys
import pandas as pd

POWER_PREF = ["InvPDC_kW_Avg", "PwrMtrP_kW_Avg", "InvPAC_kW_Avg"]
IRR_PREF   = ["RefCell1_Wm2_Avg", "SEWSPOAIrrad_Wm2_Avg", "Pyra1_Wm2_Avg", "Pyra2_Wm2_Avg"]
TEMP_PREF  = ["AmbTemp_C_Avg", "SEWSAmbientTemp_C_Avg"]
WIND_PREF  = ["WindSpeedAve_ms", "WindSpeed_ms_Max"]


def _pick(cols, prefs):
    return next((c for c in prefs if c in cols), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="unzipped NIST folder (contains year/month/*.csv)")
    ap.add_argument("--out", required=True, help="output CSV (DKASC-format)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.csv"), recursive=True))
    if not files:
        sys.exit(f"No CSV files found under {args.folder}")
    print(f"[nist] found {len(files)} daily files")

    frames, chosen = [], None
    for f in files:
        try:
            d = pd.read_csv(f, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
        if d.empty:
            continue
        if chosen is None:
            tcol = "TIMESTAMP" if "TIMESTAMP" in d.columns else d.columns[0]
            pcol = _pick(d.columns, POWER_PREF)
            icol = _pick(d.columns, IRR_PREF)
            xcol = _pick(d.columns, TEMP_PREF)
            wcol = _pick(d.columns, WIND_PREF)
            chosen = (tcol, pcol, icol, xcol, wcol)
            print(f"[nist] time={tcol} power={pcol} irradiance={icol} temp={xcol} wind={wcol}")
            if pcol is None or icol is None:
                sys.exit("[nist] could not find power/irradiance columns -- paste the header to me.")
        tcol, pcol, icol, xcol, wcol = chosen
        out = pd.DataFrame()
        out["timestamp"] = d[tcol]
        out["Active_Power"] = pd.to_numeric(d[pcol], errors="coerce")
        out["Global_Horizontal_Radiation"] = pd.to_numeric(d[icol], errors="coerce")
        if xcol: out["Weather_Temperature_Celsius"] = pd.to_numeric(d[xcol], errors="coerce")
        if wcol: out["Wind_Speed"] = pd.to_numeric(d[wcol], errors="coerce")
        frames.append(out)

    df = pd.concat(frames, ignore_index=True)
    # NIST writes -999 (and similar large negatives) as a missing/error flag.
    # Null those out BEFORE anything downstream so they can't corrupt hourly means.
    for c in ["Active_Power", "Global_Horizontal_Radiation",
              "Weather_Temperature_Celsius", "Wind_Speed"]:
        if c in df.columns:
            df.loc[df[c] <= -900, c] = float("nan")
    df = df[df["Active_Power"].notna() & (df["Active_Power"] >= 0)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[nist] wrote {len(df):,} rows -> {args.out}")
    m = df["Global_Horizontal_Radiation"] > 20
    corr = df.loc[m, ["Global_Horizontal_Radiation", "Active_Power"]].corr().iloc[0, 1]
    print(f"[nist] SANITY corr(irradiance, power) = {corr:.3f}  (want ~0.95+)")


if __name__ == "__main__":
    main()
