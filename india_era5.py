"""
ERA5 hourly irradiance for the monsoon-context comparison (paper Section V-H),
from the Open-Meteo archive (ERA5 reanalysis; data licensed CC BY 4.0,
attribution: Open-Meteo / Copernicus ERA5). One request per site for calendar
year 2023, cached to data/era5/era5_<site>_2023.json, which india_context.py
reads. No API key needed.

Run:  python3 india_era5.py            (add --force to re-download)
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "era5"
YEAR = 2023
URL = "https://archive-api.open-meteo.com/v1/archive"
SITES = [
    ("AliceSprings_SOURCE", -23.7620, 133.8740),
    ("Gaithersburg_SOURCE", 39.1434, -77.2136),
    ("Mumbai_TARGET", 19.0760, 72.8777),
    ("Jodhpur_TARGET", 26.2389, 73.0243),
    ("Chennai_TARGET", 13.0827, 80.2707),
]


def fetch(name: str, lat: float, lon: float, force: bool = False):
    path = OUT / f"era5_{name}_{YEAR}.json"
    if path.exists() and not force:
        return path, "cache"
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": f"{YEAR}-01-01", "end_date": f"{YEAR}-12-31",
        "hourly": "shortwave_radiation,temperature_2m",
        "timezone": "auto",
    })
    req = urllib.request.Request(URL + "?" + q, headers={"User-Agent": "ReSearch_PV"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    path.write_text(json.dumps(d))
    time.sleep(1.0)
    return path, "live"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lat, lon in SITES:
        p, src = fetch(name, lat, lon, args.force)
        n = len(json.loads(p.read_text())["hourly"]["time"])
        print(f"  {name:22s} {n} hourly rows ({src}) -> {p.relative_to(HERE)}")
