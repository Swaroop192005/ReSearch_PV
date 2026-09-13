"""
PVGIS 5.3 seriescalc fetcher for the zone grid.

One request per point covers YEARS[0]..YEARS[1] inclusive. Omit `raddatabase`
(PVGIS-SARAH3 has no India coverage -> HTTP 400; PVGIS-ERA5 is auto-selected).
Cached to data/pvgis_cache/pvgis_<lat>_<lon>_<y0>_<y1>.json.

MODELLED: PVGIS runs a physics simulation on ERA5 reanalysis. `irradiance` is
G(i), in-plane at the configured tilt (plane-of-array), not GHI.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import india_dataset as ds
from india_zones import YEARS, all_points

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "pvgis_cache"
PVGIS = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
PV_CONFIG = {"pvcalculation": 1, "peakpower": 1, "loss": 14, "angle": 20, "aspect": 0}


def fetch(lat: float, lon: float) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"pvgis_{lat}_{lon}_{YEARS[0]}_{YEARS[1]}.json"
    if key.exists():
        return json.loads(key.read_text())
    params = {"lat": lat, "lon": lon, "startyear": YEARS[0], "endyear": YEARS[1],
              "outputformat": "json", **PV_CONFIG}
    url = PVGIS + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ReSearch_PV"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    key.write_text(json.dumps(d))
    time.sleep(1.0)
    return d


def point_frame(zone: str, name: str, lat: float, lon: float) -> pd.DataFrame:
    d = fetch(lat, lon)
    rows = d["outputs"]["hourly"]
    raw = pd.DataFrame({
        "irradiance": [r["G(i)"] for r in rows],
        "ambient_temp": [r["T2m"] for r in rows],
        "power": [r["P"] for r in rows],
    })
    db = d["inputs"]["meteo_data"]["radiation_db"]
    fin = ds.finalize(raw, site=name, zone=zone, source="modelled", normalize=True)
    print(f"  {zone:22} {name:20} {len(rows):6} h -> {len(fin):6} daylight  "
          f"db={db}  P_max={raw['power'].max():.0f}")
    return fin


def build(out: str = str(HERE / "data" / "pv_india_pvgis.csv")) -> pd.DataFrame:
    print(f"[pvgis] zone grid, {YEARS[0]}-{YEARS[1]}, config={PV_CONFIG}")
    frames = [point_frame(z, n, la, lo) for z, n, la, lo in all_points()]
    df = pd.concat(frames, ignore_index=True)
    Path(out).parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[pvgis] {len(df):,} daylight rows across "
          f"{df['site'].nunique()} points / {df['zone'].nunique()} zones -> {out}")
    return df


if __name__ == "__main__":
    build()
