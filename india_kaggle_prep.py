"""
Kaggle Plant 1 -> measured Indian anchor, project schema.

Sum inverters to plant AC power, merge with the
plant sensor (IRRADIATION x1000 -> W/m^2, AMBIENT_TEMPERATURE), hourly resample,
capacity-factor normalize. QC gate: keep a plant only if
corr(irradiance, plant power) >= 0.90 (Plant 2 fails at 0.83).

Plant 1 has no published lat/lon, so it gets zone "kaggle_plant" and is used as
the measured test anchor, not tied to a climate zone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import india_dataset as ds

PLANTS = {
    "KagglePlant1": ("Plant_1_Generation_Data.csv", "Plant_1_Weather_Sensor_Data.csv"),
    "KagglePlant2": ("Plant_2_Generation_Data.csv", "Plant_2_Weather_Sensor_Data.csv"),
}
QC_MIN_CORR = 0.90


def _dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s, errors="coerce")
    return out


def plant_frame(folder: Path, name: str, gen_f: str, wx_f: str,
                keep_all: bool = False) -> pd.DataFrame | None:
    gen = pd.read_csv(folder / gen_f)
    wx = pd.read_csv(folder / wx_f)
    gen["ts"] = _dt(gen["DATE_TIME"])
    wx["ts"] = _dt(wx["DATE_TIME"])
    n_inv = gen["SOURCE_KEY"].nunique()
    frac_partial = float((gen.groupby("ts")["SOURCE_KEY"].nunique() < n_inv).mean())

    ac = gen.groupby("ts")["AC_POWER"].sum().rename("power")
    w = (wx.set_index("ts")[["IRRADIATION", "AMBIENT_TEMPERATURE"]]
         .rename(columns={"IRRADIATION": "irradiance", "AMBIENT_TEMPERATURE": "ambient_temp"}))
    df = pd.concat([ac, w], axis=1, sort=True).resample("1h").mean()
    df["irradiance"] = df["irradiance"] * 1000.0

    day = df[df["irradiance"] > ds.DAYLIGHT_MIN_GHI].dropna(subset=["irradiance", "power"])
    qc = float(day["power"].corr(day["irradiance"]))
    print(f"  {name}: {len(df)} hourly rows, QC corr(irr,power)={qc:.3f}, "
          f"partial-inverter timestamps={frac_partial*100:.0f}%")
    if qc < QC_MIN_CORR and not keep_all:
        print(f"    -> EXCLUDED (irradiance does not track power)")
        return None
    return ds.finalize(df.reset_index(drop=True), site=name,
                       zone="kaggle_plant", source="measured", normalize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/kaggle")
    ap.add_argument("--out", default="data/pv_india_measured.csv")
    ap.add_argument("--keep-all", action="store_true")
    args = ap.parse_args()
    folder = Path(args.folder)
    frames = [plant_frame(folder, n, g, w, args.keep_all)
              for n, (g, w) in PLANTS.items()]
    frames = [f for f in frames if f is not None]
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\n[kaggle] {len(out):,} rows -> {args.out}")
    print(ds.describe(out).to_string())


if __name__ == "__main__":
    main()
