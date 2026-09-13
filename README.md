# Cross-Site Generalization of ML Models for PV Power Prediction

Code and reproduction pipeline for:

> **Generalization of Machine-Learning Models for Solar Photovoltaic Power
> Prediction Across Technology, Climate, and Time: A Transfer-Learning Study on
> Three Continents**
> Swaroop Naik, Aditi Gawari, Ayush Mayekar, Chetan Katkar and Sneha Annappanavar
> (mentor) — Vidyalankar Institute of Technology, Mumbai, India
> *(arXiv preprint — link to be added on posting)*

ML models predict PV power with near-perfect accuracy, but almost every published
model is trained and tested on a **single site**, so nobody knows what happens
when you deploy it somewhere else. This repository measures exactly that, across
three kinds of distribution shift, on real public data from three continents:
Australia, the USA and India.

---

## Headline findings

| Shift | Mean ΔR² | Verdict |
|---|---|---|
| **Technology** (mono-Si → poly-Si → CdTe, same site) | 0.0014 | Indistinguishable from zero |
| **Time** (2017 model → 2021 data, same array) | < 0.01 | Negligible |
| **Climate** (Australia → USA, same technology) | 0.0244 | **~18× larger, p ≈ 2×10⁻¹⁰** |

After capacity-factor normalization, **only climate matters.** Technology and
multi-year drift are effectively free.

Two further results:

- **A fine-tunable model closes the climate gap; more data does not.** With 20% of
  target-site data, unsupervised alignment (CORAL, importance weighting) recovers
  ~0%, few-shot tree fine-tuning recovers 36–45%, and a fine-tuned neural encoder
  recovers **≈98%**.
- **The same failure appears in fault detection.** On GPVS-Faults, healthy-vs-faulty
  classification scores F1 = 0.999 within an operating mode but collapses across
  modes — MCC and Cohen's κ drop to ≈0.00 for MPPT→IPPT, a collapse that an F1 of
  0.88 completely hides.
- **India, a third climate (paper Section V).** Transfer into a measured Indian
  plant costs **2.1×** the Australia↔USA gap (ΔR² = 0.052, p = 6×10⁻⁶). A
  fine-tuned MLP still recovers 86–95% of it from 82 labelled hours, and a single
  gain factor fitted on 5 hours recovers up to 81%, while few-shot tree
  fine-tuning recovers ≤5%. A prior trained only on PVGIS simulations of 10
  Indian climate zones beats the foreign prior zero-shot (R² 0.959 vs 0.941).
  See [Section 7](#7-india-extension-paper-section-v).

---

## 1. Install

```bash
git clone https://github.com/Swaroop192005/ReSearch_PV.git
cd ReSearch_PV
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

CatBoost and XGBoost are optional *for running* the pipeline — the code falls back
to scikit-learn estimators if they are missing — but both are **required to
reproduce the paper's numbers** (see Section 6). On macOS, XGBoost needs OpenMP:
`brew install libomp`.

**Runs with no data at all.** Every script auto-generates a synthetic multi-site
dataset if `data/pv_multisite.csv` is absent, so you can smoke-test the whole
pipeline before downloading anything:

```bash
python3 benchmark.py --task both
```

> ⚠️ The synthetic generator exists **only** to validate that the pipeline runs.
> No number in the paper comes from synthetic data. Never report synthetic results.

---

## 2. Get the data

The three datasets are public but **not redistributable here** — download them
yourself, then run the prep scripts below.

| Dataset | Used for | Where |
|---|---|---|
| **DKASC**, Alice Springs (AU) | cross-technology + cross-time | [dkasolarcentre.com.au](https://dkasolarcentre.com.au) → per-array "full data set with weather data" |
| **NIST**, Gaithersburg MD (US) | cross-climate | [pvdata.nist.gov](https://pvdata.nist.gov) → bulk download, "Ground" array |
| **GPVS-Faults** | cross-mode fault detection | [doi:10.17632/n76t439f65](https://doi.org/10.17632/n76t439f65) |

Arrays used: Trina (5.3 kW, mono-Si), Kyocera (5.4 kW, poly-Si), Calyxo (5.4 kW,
CdTe) at Alice Springs; NIST Ground (~217 kW, mono-Si). Calendar year 2017 for the
main analysis; 2019 and 2021 for the temporal experiment.

`data/`, `results/` and `figures/` are created automatically. `data/` and
`figures/` are git-ignored; `results/` holds the committed India reference
outputs (Section 7).

---

## 3. Prepare the data

```bash
# DKASC: 3 arrays -> one hourly, capacity-factor-normalized CSV
python3 dkasc_loader.py --folder data/dkasc --out data/pv_multisite.csv \
        --start 2017-01-01 --end 2017-12-31

# NIST: stitch daily files, derive DC power, map columns to DKASC format
python3 nist_prep.py --folder ~/Downloads/nist_raw \
        --out data/dkasc/NIST_Ground_monoSi.csv
# then re-run dkasc_loader.py to combine Alice + NIST on the aligned 2017 window

# GPVS-Faults: window the time series into features
python3 gpvs_prep.py --folder ~/Downloads/gpvs/csv/CSV_Files \
        --out data/gpvs_features.csv --window 1000
```

`config.py` holds the schema (column names, feature lists, site lists). After
building a real dataset, paste the loader's printed `CONFIG SNIPPET` into it.

---

## 4. Reproduce every result

| Paper element | Command |
|---|---|
| **Table 2** within-site + zero-shot cross-site | `python3 benchmark.py --task power --train <src> --test <tgt1,tgt2>` |
| **Table 3 / Fig 3** all-pairs matrix + significance | `python3 matrix_experiment.py --task power` |
| **Table 4 / Fig 5** five-method adaptation comparison | `python3 transfer_advanced.py --task power --source Trina_monoSi --target NIST_Ground_monoSi --frac 0.2` |
| **Fig 4** few-shot recovery curve | `python3 transfer.py --task power --target NIST_Ground_monoSi` |
| **Table 5 / Fig 6** error analysis | `python3 error_analysis.py --source Trina_monoSi --target NIST_Ground_monoSi` |
| **Table 6 / Figs 7–8** cross-mode fault detection | `python3 fault_experiment.py --features data/gpvs_features.csv` |
| **Fig 9** uncertainty + OOD | `python3 uncertainty_ood.py --source Trina_monoSi --target NIST_Ground_monoSi` |
| **Fig 10** temporal generalization | `python3 temporal_experiment.py --folder data/dkasc --sites Trina_monoSi,Kyocera_polySi,Calyxo_CdTe --train-year 2017 --test-years 2019,2021` |
| **All 10 paper figures at once** | `python3 make_paper_figures.py` |

Add `--weather` to any experiment to include ambient temperature. Every
experiment with a stochastic component runs over 5 seeds (42, 1, 2, 7, 123) and
writes CSVs to `results/`.

### Figure outputs

`make_paper_figures.py` writes into `figures/paper/`. File numbers match the
paper's figure numbers exactly — Figure *N* is `figN_*.png`:

| Paper | File | Content |
|---|---|---|
| Figure 1 | `fig1_gap.png` | Cross-technology vs cross-climate gap, per model |
| Figure 2 | `fig2_withincross.png` | Within-site vs zero-shot R² on NIST |
| Figure 3 | `fig3_matrix.png` | All-pairs generalization heatmap |
| Figure 4 | `fig4_recovery.png` | Few-shot recovery vs target data used |
| Figure 5 | `fig5_methods.png` | Gap recovered by each adaptation method |
| Figure 6 | `fig6_err.png` | Cross-climate error by irradiance band |
| Figure 7 | `fig7_fault.png` | Within- vs cross-mode fault detection |
| Figure 8 | `fig8_confusion.png` | Cross-mode confusion matrices |
| Figure 9 | `fig9_coverage.png` | Conformal interval coverage, source vs target |
| Figure 10 | `fig10_threeaxes.png` | The three generalization axes compared |

---

## 5. File map

**Core** — `config.py` (schema/settings) · `data.py` (loading, splits) ·
`models.py` (6 regressors / classifiers) · `metrics.py` (R²/RMSE/MAE;
Accuracy/Precision/Recall/F1/AUC/MCC/κ) · `benchmark.py` (within-site,
cross-site) · `make_synthetic_data.py`

**Data prep** — `dkasc_loader.py` (auto column detection, hourly resample,
timezone-safe, capacity-factor normalization) · `nist_prep.py` (daily files →
DKASC format, sentinel handling) · `gpvs_prep.py` (time series → windowed features)

**Experiments** — `matrix_experiment.py` (all-pairs + CIs + significance) ·
`transfer.py` (few-shot + CORAL) · `transfer_advanced.py` (5-method comparison) ·
`error_analysis.py` (error by irradiance/season + calibration) ·
`uncertainty_ood.py` (conformal coverage + OOD detection) ·
`temporal_experiment.py` (train-year → later-year drift) ·
`fault_experiment.py` (within- vs cross-mode fault detection)

**Figures** — `make_paper_figures.py` produces the **ten figures in the paper**,
into `figures/paper/`, numbered to match. `make_figures.py` is a separate,
earlier exploratory script (degradation / RMSE / recovery / weather plots) that
writes to `figures/` and uses its own unrelated numbering — it is not the source
of any paper figure.

**India extension** — `india_*.py` and `make_india_figures.py`; see Section 7.

---

## 6. Methodological notes

- **Capacity-factor normalization** (power ÷ 99th-percentile output) is what makes
  a 5 kW array and a 217 kW array comparable. It is applied by the loader and is
  essential to every cross-site result.
- **Features are irradiance** (+ ambient temperature with `--weather`). Hour and
  month are deliberately excluded: clock hour is timezone-dependent and calendar
  month is season-inverted between hemispheres, so both would inject spurious
  cross-site artifacts. Irradiance already encodes diurnal and seasonal variation.
- **The generalization gap** is measured against the *target's* within-site
  ceiling, not the source's. Using the source's ceiling produces meaningless
  negative "drops".
- **Sensor conventions differ** between sites: DKASC reports global-horizontal
  irradiance, NIST reports plane-of-array reference-cell irradiance. This partly
  explains NIST's higher in-domain predictability and is an acknowledged confound.
- **Curtailed sites were excluded.** Installations subject to grid curtailment
  (e.g. solar+diesel microgrids) are not weather-predictable and were dropped
  during dataset selection.
- **Which model produces which result.** Six regressors are evaluated in total:
  Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost, CatBoost and
  Linear Regression. Figures 1 and 2 report **all four boosted-tree models side by
  side** (Gradient Boosting, HistGradientBoosting, XGBoost, CatBoost). In Table 2
  the per-site best model is CatBoost for three sites and **Gradient Boosting** for
  Kyocera. CatBoost then *anchors* the single-model analyses — the all-pairs matrix
  (Table 3), adaptation comparison (Table 4), error analysis (Table 5) and fault
  detection (Table 6) — because it wins the within-site comparison on most sites.
- **CatBoost must be installed to reproduce the paper's numbers.** The anchor-model
  helper falls back in the order CatBoost → XGBoost → HistGradientBoosting →
  Gradient Boosting → Random Forest. If CatBoost is missing, every result the paper
  labels "CatBoost" silently becomes an XGBoost result and your numbers will not
  match. Verify before running:

  ```bash
  python3 -c "import catboost, xgboost; print('both present')"
  ```

---

## 7. India extension (paper Section V)

Section V of the paper extends the benchmark to a third, tropical,
monsoon-driven climate. The pipeline is unchanged: CatBoost and the same MLP as
`transfer_advanced.py`, capacity-factor target, irradiance (+ ambient
temperature in the weather variant), 5 seeds, 60/40 transfer split. The
measured target is **Kaggle Plant 1**, an anonymised grid-connected Indian
plant. An **India-native prior** is trained on PVGIS simulations of 24 points in
10 Indian solar-climate zones.

> **Measured vs simulated.** Every score on the Indian plant is measured on real
> data. PVGIS output is simulated: it is used only to train the India-native
> prior and as the target of the leave-one-zone-out analysis, which the paper
> labels as simulated.

### 7.1 Data

| Dataset | Used for | Where |
|---|---|---|
| **Kaggle *Solar Power Generation Data***, Plant 1 | measured Indian target — 412 daylight hours, 15 May–17 Jun 2020 | [kaggle.com/datasets/anikannal/solar-power-generation-data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) → put the 4 CSVs in `data/kaggle/` |
| **PVGIS 5.3** (`PVGIS-ERA5`), 2019–2023 | India-native prior; leave-one-zone-out (simulated) | downloaded by `india_pvgis.py` |
| **ERA5** via the Open-Meteo archive, 2023 | monsoon irradiance context (Section V-H) | downloaded by `india_era5.py` |

None of these is redistributed here.

### 7.2 Build the India data

Run from the repository root, after building `data/pv_multisite.csv` (Section 3):

```bash
python3 india_kaggle_prep.py
python3 india_pvgis.py
python3 india_era5.py
```

- `india_kaggle_prep.py` sums the 22 inverters, resamples to hourly, converts
  `IRRADIATION` to W/m² (×1000) and applies the quality gate — Plant 1 passes,
  Plant 2 is excluded because its output tracks inverter dropouts rather than
  the weather → `data/pv_india_measured.csv`.
- `india_pvgis.py` makes one PVGIS request per point (24 in total, cached in
  `data/pvgis_cache/`) → `data/pv_india_pvgis.csv`, 500,403 daylight hours.
- `india_era5.py` downloads 2023 hourly GHI for Alice Springs, Gaithersburg,
  Mumbai, Jodhpur and Chennai → `data/era5/`.

The India comparisons also read the AU↔US reference matrices:

```bash
python3 matrix_experiment.py --task power
python3 matrix_experiment.py --task power --weather
```

### 7.3 Reproduce Section V

| Paper element | Command |
|---|---|
| **Fig 11** gap to/from India vs AU↔US, significance tests | `python3 india_matrix.py` |
| **Table 7** five-method adaptation with India as target | `python3 india_transfer.py` |
| **Fig 13** recovery vs labelled Indian hours (gain / tree / MLP) | `python3 india_recovery.py` |
| **Table 8 / Fig 14** India error by irradiance | `python3 india_error.py` |
| **Fig 12** India-native vs AU+US prior; **Table 9** leave-one-zone-out | `python3 india_native.py` (slow: trains on 500k rows) |
| Plant 1/2 quality gate, PVGIS physics check (V-F), monsoon context (V-H) | `python3 india_context.py` |
| **Figs 11–14** | `python3 make_india_figures.py` → `figures/india/` |

| Paper | File |
|---|---|
| Figure 11 | `fig11_india_gap.png` |
| Figure 12 | `fig14_india_prior.png` |
| Figure 13 | `fig12_india_recovery.png` |
| Figure 14 | `fig13_india_err.png` |

File names follow the order the figures were produced; LaTeX float placement
numbers them differently in the paper.

### 7.4 Reference outputs (committed)

The exact outputs behind Section V are committed in `results/`, so every India
number in the paper can be checked without rerunning anything:

- `india_matrix_raw.csv`, `india_transfer_raw.csv`, `india_recovery_raw.csv`,
  `india_error_by_irradiance.csv`, `india_native_raw.csv`,
  `india_lozo_catboost.csv` — every seed and every method.
- `matrix_raw_power_nw.csv`, `matrix_raw_power_w.csv` — the AU↔US reference the
  India gaps are compared against.
- `india_jobA.log`, `india_jobB.log`, `india_context.log` — full console output.
  Step A5 in `india_jobA.log` ends in a traceback from a since-fixed bug in
  `india_context.py`; `india_context.log` is the complete rerun with the fixed
  script.

Rerunning a script overwrites its CSV, and `git diff results/` then shows
whether your run matches.

Environment used: Python 3.14.2, scikit-learn 1.9.0, CatBoost 1.2.10, pandas
3.0.5, NumPy 2.5.2, SciPy 1.18.1, Matplotlib 3.11.2 (Apple silicon, CPU only).
The same environment reproduces the published Table 3 and Table 4 numbers
exactly, so the India results are directly comparable with them.

---

## 8. Citation

```bibtex
@article{naik2026pvgeneralization,
  title   = {Generalization of Machine-Learning Models for Solar Photovoltaic
             Power Prediction Across Technology, Climate, and Time:
             A Transfer-Learning Study on Three Continents},
  author  = {Naik, Swaroop and Gawari, Aditi and Mayekar, Ayush and
             Katkar, Chetan and Annappanavar, Sneha},
  journal = {arXiv preprint},
  year    = {2026}
}
```

Please also cite the underlying datasets (DKASC, NIST, GPVS-Faults) — see the
links in Section 2.

---

## 9. License

Code in this repository is released under the **MIT License** — see [`LICENSE`](LICENSE).

The three datasets are **not** covered by that licence and remain under their own
respective terms; obtain them from the sources in Section 2 and cite them
accordingly. The same applies to the Kaggle, PVGIS and ERA5 data used by the
India extension (Section 7); Open-Meteo/ERA5 data are CC BY 4.0 and require
attribution.
