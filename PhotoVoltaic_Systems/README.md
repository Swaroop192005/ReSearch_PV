# Generalization of ML Models for Solar PV Power Prediction — Code & Reproducibility

Code for the study **"Generalization of Machine-Learning Models for Solar Photovoltaic
Power Prediction Across Technology, Climate, and Time: A Transfer-Learning Study on Two
Continents."**

It measures how well ML models for PV power prediction transfer across three kinds of
distribution shift — **panel technology**, **geographic climate**, and **elapsed time** —
plus a cross-domain **fault-detection** experiment. Headline finding: after capacity-factor
normalization, **only climate materially degrades generalization**; technology and multi-year
time drift are essentially free, and a fine-tuned neural encoder recovers ~98% of the climate gap.

---

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

CatBoost/XGBoost are optional (the code falls back to scikit-learn if they're missing).
On macOS, XGBoost needs OpenMP: `brew install libomp`.

**Runs immediately with no data:** every script auto-generates a synthetic multi-site dataset
if `data/pv_multisite.csv` is absent, so you can smoke-test the whole pipeline before downloading anything.

```bash
python3 benchmark.py --task both      # synthetic data, both tasks
```

---

## 2. Datasets (public)

| Dataset | Use | Source |
|---|---|---|
| **DKASC**, Alice Springs (AU) | cross-technology + cross-time | dkasolarcentre.com.au → per-array "full data set with weather data" |
| **NIST**, Gaithersburg MD (US) | cross-climate | pvdata.nist.gov (bulk download, "Ground" array) |
| **GPVS-Faults** | cross-mode fault detection | data.mendeley.com/datasets/n76t439f65 |

Download, then build the combined power dataset with the loaders below. `data/`, `results/`,
and `figures/` are created automatically and are **not** shipped in this archive.

---

## 3. Data preparation

```bash
# DKASC: 3 arrays (mono/poly/CdTe) -> one hourly, capacity-factor-normalized CSV
python3 dkasc_loader.py --folder data/dkasc --out data/pv_multisite.csv --start 2017-01-01 --end 2017-12-31

# NIST: stitch the daily files, derive DC power, map columns to DKASC format
python3 nist_prep.py --folder ~/Downloads/nist_raw --out data/dkasc/NIST_Ground_monoSi.csv
# then re-run dkasc_loader.py on data/dkasc to combine Alice + NIST (aligned 2017 window)

# GPVS-Faults: window the time-series into features (fault=0 for F0, 1 for F1-F7; mode from filename)
python3 gpvs_prep.py --folder ~/Downloads/gpvs/csv/CSV_Files --out data/gpvs_features.csv --window 1000
```

`config.py` holds the schema (column names, feature lists, site lists). After building a real
dataset, paste the loader's printed CONFIG SNIPPET into `config.py`.

---

## 4. Reproduce each result

| Paper element | Command |
|---|---|
| **Table 1** within-site + **O2** cross-site | `python3 benchmark.py --task power --train <src> --test <tgt1,tgt2>` |
| **Table 2 / Fig 3** all-pairs matrix + significance | `python3 matrix_experiment.py --task power` (add `--weather`) |
| **Fig 1/2** gap & within-vs-cross | derived from `benchmark.py` outputs; `make_figures.py` |
| **Table 3 / Fig 4-5** 5-method adaptation comparison | `python3 transfer_advanced.py --task power --source Trina_monoSi --target NIST_Ground_monoSi --frac 0.2` |
| few-shot recovery curve | `python3 transfer.py --task power --target NIST_Ground_monoSi` |
| **Sec 4.6 / Fig 6** error analysis | `python3 error_analysis.py --source Trina_monoSi --target NIST_Ground_monoSi` |
| **Sec 4.7 / Fig 7** cross-mode fault detection | `python3 fault_experiment.py --features data/gpvs_features.csv` |
| **Sec 4.8 / Fig 8** uncertainty & OOD | `python3 uncertainty_ood.py --source Trina_monoSi --target NIST_Ground_monoSi` |
| **Sec 4.9 / Fig 9** temporal generalization | `python3 temporal_experiment.py --folder data/dkasc --sites Trina_monoSi,Kyocera_polySi,Calyxo_CdTe --train-year 2017 --test-years 2019,2021` |
| paper figures from result CSVs | `python3 make_figures.py` |

Add `--weather` to any experiment to include ambient temperature. All experiments use 5 random
seeds where applicable and write CSVs to `results/`.

---

## 5. File map

**Core:** `config.py` (schema/settings) · `data.py` (loading, splits) · `models.py` (6 regressors /
classifiers) · `metrics.py` · `benchmark.py` (O1 within-site, O2 cross-site) · `make_synthetic_data.py`.

**Data prep:** `dkasc_loader.py` (DKASC → capacity-factor CSV; auto column detection, hourly
resample, tz-safe) · `nist_prep.py` (NIST daily files → DKASC format; sentinel handling) ·
`gpvs_prep.py` (GPVS-Faults time-series → windowed features).

**Experiments:** `matrix_experiment.py` (all-pairs + CIs + significance) · `transfer.py` (few-shot +
CORAL) · `transfer_advanced.py` (5-method comparison incl. importance weighting + MLP fine-tuning) ·
`error_analysis.py` (error by irradiance/season + calibration) · `uncertainty_ood.py` (conformal
coverage + OOD detection) · `temporal_experiment.py` (train-year → later-year drift) ·
`fault_experiment.py` (within- vs cross-mode fault detection).

**Figures:** `make_figures.py`.

---

## 6. Notes

- **Capacity-factor normalization** (power ÷ 99th-percentile) is what makes arrays of 5 kW and
  217 kW comparable; it is applied by the loader and is essential for cross-site work.
- **Features** are irradiance (+ ambient temperature with `--weather`). Hour and month are excluded
  for cross-hemisphere comparability; irradiance already encodes diurnal/seasonal variation.
- **Synthetic data** exists only to validate the pipeline — never report synthetic numbers.
- Reported results use **CatBoost**; the boosted-tree family gives consistent results.
