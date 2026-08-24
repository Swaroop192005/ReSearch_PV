"""
Synthetic multi-site PV data generator.

Lets you RUN the pipeline before downloading DKASC / GPVS-Faults. Each "site" is
given deliberately different characteristics (sensor scaling, temperature, fault
rate, noise) so cross-site degradation (O2) is real and measurable. When your
real data is ready, ignore this file and point config.DATA_CSV at your CSV with
the same column names.
"""
import numpy as np
import pandas as pd

import config


def _generate_site(site_name, n, rng, profile):
    irradiance = np.clip(rng.beta(2.0, 2.5, n) * profile["irr_scale"], 0, 1200)
    ambient_temp = rng.normal(profile["amb_mean"], 6, n)
    humidity = np.clip(rng.normal(profile["hum_mean"], 15, n), 5, 100)
    inverter_temp = 15 + 0.03 * irradiance + 0.5 * ambient_temp + rng.normal(0, 2, n)

    # --- Faults occur first, then physically suppress the measured currents ---
    fault_logit = profile["fault_bias"] + 0.05 * (inverter_temp - 30) + rng.normal(0, 0.4, n)
    fault_prob = 1 / (1 + np.exp(-fault_logit))
    fault = rng.binomial(1, np.clip(fault_prob, 0, 0.95), n)
    fault_factor = np.where(fault == 1, rng.uniform(0.10, 0.45, n), 1.0)

    base_i = irradiance / 100.0 * profile["i_gain"] * fault_factor
    i_mppt1 = np.clip(base_i + rng.normal(0, 0.15, n), 0, 13)
    i_mppt2 = np.clip(base_i * 0.97 + rng.normal(0, 0.15, n), 0, 13)
    i_mppt3 = np.clip(base_i * 1.01 + rng.normal(0, 0.15, n), 0, 13)

    v_mppt1 = np.clip(rng.normal(profile["v_mean"], 30, n), 0, 400)
    v_mppt2 = np.clip(rng.normal(profile["v_mean"] * 0.98, 32, n), 0, 400)
    v_mppt3 = np.clip(rng.normal(profile["v_mean"] * 1.06, 38, n), 0, 420)
    voltage_ua = np.clip(rng.normal(230, 20, n), 0, 260)
    current_ica1 = np.clip(base_i * 3.3 + rng.normal(0, 0.5, n), 0, 42)
    frequency = np.clip(rng.normal(50, 0.15, n), 48, 51)

    power = (
        (v_mppt1 * i_mppt1 + v_mppt2 * i_mppt2 + v_mppt3 * i_mppt3)
        * (1 - 0.004 * np.clip(inverter_temp - 25, 0, None))
        * profile["p_gain"]
    )
    power = np.clip(power + rng.normal(0, 25, n), 0, None)

    return pd.DataFrame({
        config.SITE_COL: site_name,
        "v_mppt1": v_mppt1, "v_mppt2": v_mppt2, "v_mppt3": v_mppt3,
        "i_mppt1": i_mppt1, "i_mppt2": i_mppt2, "i_mppt3": i_mppt3,
        "voltage_ua": voltage_ua, "current_ica1": current_ica1,
        "frequency": frequency, "inverter_temp": inverter_temp,
        "irradiance": irradiance, "ambient_temp": ambient_temp, "humidity": humidity,
        config.POWER_COL: power, config.FAULT_COL: fault.astype(int),
    })


SITE_PROFILES = {
    "A": dict(irr_scale=1000, amb_mean=25, hum_mean=45, i_gain=1.00, v_mean=265, p_gain=1.00, fault_bias=-2.3),
    "B": dict(irr_scale=900,  amb_mean=32, hum_mean=60, i_gain=1.15, v_mean=240, p_gain=1.10, fault_bias=-1.8),
    "C": dict(irr_scale=1100, amb_mean=18, hum_mean=35, i_gain=0.90, v_mean=300, p_gain=0.92, fault_bias=-2.7),
}


def generate(n_per_site=20000, seed=config.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    frames = [_generate_site(s, n_per_site, rng, SITE_PROFILES[s]) for s in config.ALL_SITES]
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(config.DATA_CSV, index=False)
    print(f"Wrote {len(df):,} rows to {config.DATA_CSV}")
    print(df.groupby(config.SITE_COL)[[config.POWER_COL, config.FAULT_COL]].mean())
