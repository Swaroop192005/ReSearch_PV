"""
Context numbers for the India section, recomputed from raw data for provenance:
  (a) Kaggle Plant 1 vs Plant 2 quality gate
  (b) PVGIS physics (Faiman thermal + Huld power model) vs Plant 1's own sensors
  (c) ERA5 2023 irradiance climate: share of daylight hours / energy below 400 W/m^2

Run:  python india_context.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
KAG = HERE / "data" / "kaggle"
ERA5 = HERE / "data" / "era5"  # written by india_era5.py


def dt(s):
    out = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
    return out if out.isna().mean() < 0.5 else pd.to_datetime(s, errors="coerce")


# ---------------- (a) quality gate ----------------
print("===== (a) KAGGLE QUALITY GATE: corr(irradiance, plant power), daylight hourly =====")
plants = {}
for p in (1, 2):
    g = pd.read_csv(KAG / f"Plant_{p}_Generation_Data.csv")
    w = pd.read_csv(KAG / f"Plant_{p}_Weather_Sensor_Data.csv")
    g["ts"], w["ts"] = dt(g["DATE_TIME"]), dt(w["DATE_TIME"])
    n_inv = g["SOURCE_KEY"].nunique()
    partial = float((g.groupby("ts")["SOURCE_KEY"].nunique() < n_inv).mean())
    ac = g.groupby("ts")["AC_POWER"].sum().rename("P")
    wx = w.set_index("ts")[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]]
    d = pd.concat([ac, wx], axis=1, sort=True)
    hr = d.resample("1h").mean()
    hr = hr[hr["IRRADIATION"] * 1000 > 20].dropna()
    corr = hr["P"].corr(hr["IRRADIATION"])
    plants[p] = d
    print(f"  Plant {p}: {n_inv} inverters, {len(hr)} daylight h, {g['ts'].min().date()} -> "
          f"{g['ts'].max().date()}, corr={corr:.3f}, timestamps missing >=1 inverter={100 * partial:.0f}%")

# ---------------- (b) PVGIS physics vs Plant 1 ----------------
print("\n===== (b) PVGIS PV MODEL vs KAGGLE PLANT 1 (15-min records) =====")
d = plants[1].dropna().copy()
d["G"] = d["IRRADIATION"] * 1000.0
d = d[d["G"] > 20]
Ta, G, Tm = d["AMBIENT_TEMPERATURE"].values, d["G"].values, d["MODULE_TEMPERATURE"].values
U_pv = 26.9 + 6.20 * 1.5
U_fit = float(np.sum(G * G) / np.sum(G * (Tm - Ta)))
print(f"  records={len(d)}  Faiman U (PVGIS c-Si, 1.5 m/s)={U_pv:.1f}  best-fit U={U_fit:.1f} "
      f"({100 * abs(U_fit - U_pv) / U_pv:.1f}% apart)")
print(f"  Faiman vs module sensor: R2={r2_score(Tm, Ta + G / U_pv):.3f}  "
      f"bias={np.mean(Ta + G / U_pv - Tm):+.2f} C")
k = [-0.017237, -0.040465, -0.004702, 0.000149, 0.000170, 0.000005]


def huld(G, T):
    Gp = np.clip(G, 1e-6, None) / 1000.0; Tp = T - 25.0; lg = np.log(Gp)
    return (G / 1000.0) * (1 + k[0] * lg + k[1] * lg ** 2 + k[2] * Tp + k[3] * Tp * lg
                           + k[4] * Tp * lg ** 2 + k[5] * Tp ** 2)


cf = np.clip(d["P"].values / np.quantile(d["P"].values, 0.99), 0, 1.2)
for lab, T in (("measured module temp", Tm), ("Faiman-estimated temp", Ta + G / U_fit)):
    P = huld(G, T); cfp = np.clip(P / np.quantile(P, 0.99), 0, 1.2)
    print(f"  Huld driven by {lab:22s}: R2={r2_score(cf, cfp):.4f}  bias={np.mean(cfp - cf):+.4f} CF")

# ---------------- (c) irradiance climate ----------------
print("\n===== (c) ERA5 2023 DAYLIGHT IRRADIANCE CLIMATE (GHI > 5 W/m^2) =====")
SITES = [("Alice Springs (source)", "AliceSprings_SOURCE"), ("Gaithersburg (source)", "Gaithersburg_SOURCE"),
         ("Mumbai", "Mumbai_TARGET"), ("Jodhpur", "Jodhpur_TARGET"), ("Chennai", "Chennai_TARGET")]
print(f"  {'site':24s}{'annual <400 h':>14s}{'<400 energy':>13s}{'mean GHI':>10s}"
      f"{'JJAS <400 h':>13s}{'JJAS GHI':>10s}{'OND <400 h':>12s}")
for label, key in SITES:
    h = json.load(open(ERA5 / f"era5_{key}_2023.json"))["hourly"]
    t = pd.to_datetime(h["time"])
    g = np.array([np.nan if v is None else v for v in h["shortwave_radiation"]], float)
    day = g > 5
    def share(mask):
        x = g[mask & day]
        return 100 * np.mean(x < 400), 100 * x[x < 400].sum() / x.sum(), x.mean()
    a = share(np.ones_like(day)); j = share(np.asarray(t.month.isin([6, 7, 8, 9])))
    o = share(np.asarray(t.month.isin([10, 11, 12])))
    print(f"  {label:24s}{a[0]:>13.1f}%{a[1]:>12.1f}%{a[2]:>10.0f}{j[0]:>12.1f}%{j[2]:>10.0f}{o[0]:>11.1f}%")
