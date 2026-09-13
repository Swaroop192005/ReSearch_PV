"""
Indian solar-climate zones and the representative points sampled from each.

Ten zones chosen for how they load a PV model differently: irradiance level,
which monsoon and when, humidity/haze, altitude. 2-3 points per zone give
within-zone spread for leave-one-zone-out validation.
"""
from __future__ import annotations

# zone -> list of (name, lat, lon)
ZONES: dict[str, list[tuple[str, float, float]]] = {
    "thar_arid": [
        ("Jaisalmer", 26.9157, 70.9083),
        ("Bikaner", 28.0229, 73.3119),
        ("Jodhpur", 26.2389, 73.0243),
    ],
    "deccan_semiarid": [
        ("Pune", 18.5204, 73.8567),
        ("Solapur", 17.6599, 75.9064),
        ("Ballari", 15.1394, 76.9214),
    ],
    "konkan_sw_monsoon": [
        ("Mumbai", 19.0760, 72.8777),
        ("Panaji", 15.4909, 73.8278),
        ("Mangaluru", 12.9141, 74.8560),
    ],
    "coromandel_ne_monsoon": [
        ("Chennai", 13.0827, 80.2707),
        ("Nellore", 14.4426, 79.9865),
    ],
    "gangetic_plain": [
        ("Delhi", 28.6139, 77.2090),
        ("Lucknow", 26.8467, 80.9462),
        ("Patna", 25.5941, 85.1376),
    ],
    "eastern_humid": [
        ("Kolkata", 22.5726, 88.3639),
        ("Bhubaneswar", 20.2961, 85.8245),
    ],
    "northeast_hills": [
        ("Guwahati", 26.1445, 91.7362),
        ("Shillong", 25.5788, 91.8933),
    ],
    "trans_himalaya": [
        ("Leh", 34.1526, 77.5771),
        ("Kaza", 32.2270, 78.0716),
    ],
    "central_highlands": [
        ("Bhopal", 23.2599, 77.4126),
        ("Nagpur", 21.1458, 79.0882),
    ],
    "southern_peninsula": [
        ("Thiruvananthapuram", 8.5241, 76.9366),
        ("Madurai", 9.9252, 78.1198),
    ],
}

YEARS = (2019, 2023)  # PVGIS startyear, endyear (inclusive)


def all_points() -> list[tuple[str, str, float, float]]:
    """(zone, name, lat, lon) for every point."""
    return [(z, n, la, lo) for z, pts in ZONES.items() for (n, la, lo) in pts]
