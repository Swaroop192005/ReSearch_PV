"""
Shared helpers for extending the paper's pipeline to India.

Every model setting mirrors the paper exactly (models.py and transfer_advanced.py
in this repository), and the same environment reproduces the published AU<->US
numbers bit-for-bit, so India results are directly comparable to them.

Inputs (see README, section 9):
  data/pv_multisite.csv      AU/US arrays    (dkasc_loader.py + nist_prep.py)
  data/pv_india_measured.csv Kaggle Plant 1  (india_kaggle_prep.py)
  data/pv_india_pvgis.csv    PVGIS grid      (india_pvgis.py)
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MULTISITE = HERE / "data" / "pv_multisite.csv"
INDIA_CSV = HERE / "data" / "pv_india_measured.csv"
PVGIS_CSV = HERE / "data" / "pv_india_pvgis.csv"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

INDIA = "KagglePlant1_India"
AUUS = ["Calyxo_CdTe", "Kyocera_polySi", "NIST_Ground_monoSi", "Trina_monoSi"]
SEEDS = [42, 1, 2, 7, 123]


def feats(weather: bool) -> list[str]:
    return ["irradiance", "ambient_temp"] if weather else ["irradiance"]


def load_auus() -> pd.DataFrame:
    return pd.read_csv(MULTISITE)[["site", "power", "irradiance", "ambient_temp"]]


def load_india() -> pd.DataFrame:
    """Kaggle Plant 1: hourly, daylight-only, power already capacity-factor
    normalised by its own 99th percentile -- the paper's preprocessing."""
    d = pd.read_csv(INDIA_CSV)
    d = d[d["site"] == "KagglePlant1"][["power", "irradiance", "ambient_temp"]].copy()
    d.insert(0, "site", INDIA)
    return d.reset_index(drop=True)


def load_all() -> pd.DataFrame:
    return pd.concat([load_auus(), load_india()], ignore_index=True)


def xy(df: pd.DataFrame, sites: list[str], weather: bool):
    sub = df[df["site"].isin(sites)]
    X = sub[feats(weather)].astype(float)
    X = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)
    return X, sub["power"].astype(float).values


def catboost(seed: int = 42):
    """Paper setting: models.py -> CatBoostRegressor(600, lr 0.05, depth 6)."""
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6,
                             random_state=seed, verbose=0)


def paper_mlp_pretrained(Xs, ys):
    """Paper setting: transfer_advanced.py -> MLP(32,16), 400 iters, warm start,
    scaler fitted on the source only."""
    sc = StandardScaler().fit(Xs)
    net = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=400,
                       random_state=0, warm_start=True)
    net.fit(sc.transform(Xs), ys)
    return sc, net


def paper_mlp_finetune(sc, net, Xk, yk):
    """Paper setting: net.max_iter = 200; net.fit(target subset)."""
    n = copy.deepcopy(net)
    n.max_iter = 200
    n.fit(sc.transform(Xk), yk)
    return n
