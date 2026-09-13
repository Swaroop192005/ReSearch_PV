"""
Dataset layer for the Indian data (Kaggle Plant 1 and the PVGIS grid).
Same preprocessing as the paper: daylight filter, capacity-factor target.

  FEATURES = irradiance (W/m^2) + ambient temperature (deg C). Nothing else.
  TARGET   = capacity factor = power / its own 99th percentile, clipped [0, 1.2].

Every loader returns columns: site, zone, source, power, irradiance, ambient_temp
where `power` is already the capacity factor and `source` is "measured" or
"modelled".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["irradiance", "ambient_temp"]
TARGET = "power"
SITE = "site"
ZONE = "zone"
SOURCE = "source"
SCHEMA = [SITE, ZONE, SOURCE, TARGET, *FEATURES]

DAYLIGHT_MIN_GHI = 20.0


def capacity_factor(power_w: pd.Series) -> pd.Series:
    cap = power_w.quantile(0.99)
    if not cap or cap <= 0:
        raise ValueError("99th-percentile power is non-positive")
    return (power_w / cap).clip(lower=0.0, upper=1.2)


def finalize(df: pd.DataFrame, site: str, zone: str, source: str,
             normalize: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df[df["irradiance"].fillna(0) > DAYLIGHT_MIN_GHI]
    df = df[df["power"].notna() & (df["power"] >= 0)]
    df = df.dropna(subset=["irradiance", "ambient_temp"])
    if normalize:
        df["power"] = capacity_factor(df["power"])
    df[SITE] = site
    df[ZONE] = zone
    df[SOURCE] = source
    return df[SCHEMA].reset_index(drop=True)


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (df[FEATURES].to_numpy(dtype=float),
            df[TARGET].to_numpy(dtype=float))


def describe(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby([ZONE, SOURCE])
    return pd.DataFrame({
        "rows": g.size(),
        "cf_mean": g[TARGET].mean().round(3),
        "ghi_mean": g["irradiance"].mean().round(0),
        "temp_mean": g["ambient_temp"].mean().round(1),
    })
