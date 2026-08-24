"""
Central configuration for the PV cross-site generalization study.

Edit the column names here to match your real dataset (DKASC, GPVS-Faults, etc.).
Everything else (data loading, models, experiments) reads from this file, so you
only change column names in one place when you move from synthetic to real data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# CSV the pipeline reads. If missing, synthetic data is generated automatically.
DATA_CSV = DATA_DIR / "pv_multisite.csv"

# ---- Schema (rename to your dataset's columns when using real data) ----
SITE_COL = "site"
POWER_COL = "power"
FAULT_COL = "fault"

ELECTRICAL_FEATURES = [
    "v_mppt1", "v_mppt2", "v_mppt3",
    "i_mppt1", "i_mppt2", "i_mppt3",
    "voltage_ua", "current_ica1", "frequency",
    "inverter_temp",
]
WEATHER_FEATURES = ["irradiance", "ambient_temp", "humidity"]

# ---- Cross-site design ----
ALL_SITES = ["A", "B", "C"]
TRAIN_SITES = ["A"]
TEST_SITES = ["B", "C"]

# ---- Reproducibility ----
RANDOM_STATE = 42
TEST_SIZE = 0.2
