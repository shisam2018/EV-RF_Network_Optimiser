"""
================================================================================
 EV-RF Network Optimiser
 Optimal siting and capacity planning of Electric Vehicle Recharging
 Facilities (EV-RF) for Bengaluru, Karnataka, India
================================================================================

A single-file research + decision-support tool that implements, end to end:

  1. Synthetic-but-realistic demand/supply data generation, geo-referenced to
     Bengaluru (Bengaluru Urban / BBMP extent), matching the dataset design of
     the source study (64x64 = 4,096 demand grid points; 100 candidate EV-RF
     supply/parking locations; demand history extended through 2010-2025
     annual actuals plus a partial-year Jan-Jun 2026 observation, with the
     operational forecast horizon rolled forward to full-year 2027/2028).
  2. Multi-model demand forecasting (parabolic, cubic-polynomial, ARIMA,
     naive), each backtested out-of-sample.
  3. An exact Mixed-Integer Linear Program (MILP) for EV-RF siting and
     slow/fast charging-station capacity expansion, solved to (near-)global
     optimality with the COIN-OR CBC solver's Branch-and-Bound algorithm.
  4. Two standard heuristic baselines (nearest-assignment greedy;
     LP-relaxation + rounding) for a genuine, quantitative comparative
     analysis against the exact Branch-and-Bound solution.
  5. GeoJSON-driven interactive maps (Folium) and downloadable GeoJSON/CSV
     data products.
  6. A professional Streamlit UI tying all of the above together with KPI
     dashboards.

Run with:   streamlit run app.py
Requires:   streamlit, pandas, numpy, scipy, pulp, statsmodels, folium,
            streamlit-folium, matplotlib

Author: Prepared for Dr. Shisam Bhattacharyya, Apexon-Digital Engineering.
================================================================================


MATHEMATICAL MODEL
===================
Sets
    I : demand points (centre of each grid block),        i = 0 .. n_i-1
    J : candidate EV-RF supply (parking) locations,        j = 0 .. n_j-1
    A : allowed (i, j) service pairs, i.e. supply points within the
        maximum practical detour radius R_max of demand point i
        (a demand point outside R_max of every supply point cannot be
        served and is treated as structurally unmet demand).

Parameters
    D_i            forecast EV recharging demand at demand point i
    PS_j           total parking slots available at supply point j
    SCS0_j, FCS0_j existing slow/fast charging stations at j as of 2018
    Cap_SCS = 200  charging capacity of one slow charging station
    Cap_FCS = 400  charging capacity of one fast charging station
    r       = 1.5  relative capital/O&M cost of an FCS vs an SCS
    Dist_ij        haversine (great-circle) distance, km, for (i,j) in A
    a, b, c        cost weights (a = 1, b = 25, c = 600, per the source study)
    M_unmet        large penalty cost per unit of unmet demand, used to drive
                   the model towards maximal feasible coverage before trading
                   off distance/infrastructure cost

Decision variables
    SCS_j >= SCS0_j , integer      slow charging stations installed at j
    FCS_j >= FCS0_j , integer      fast charging stations installed at j
    DS_ij >= 0 , continuous, (i,j) in A     demand of i served by j
    U_i   >= 0 , continuous                 unmet demand at i

Objective
    min  a * sum_{(i,j) in A} Dist_ij * DS_ij      ... Cost_CD (customer dissatisfaction)
       + c * sum_j (SCS_j + r * FCS_j)             ... Cost_IF (infrastructure)
       + M_unmet * sum_i U_i                       (coverage-forcing penalty)

Constraints
    (1) sum_{j: (i,j) in A} DS_ij + U_i = D_i                for all i     (demand balance)
    (2) sum_{i: (i,j) in A} DS_ij <= Cap_SCS*SCS_j + Cap_FCS*FCS_j  for all j (capacity)
    (3) SCS_j + FCS_j <= PS_j                                for all j     (parking-slot limit)
    (4) SCS_j >= SCS0_j ,  FCS_j >= FCS0_j                    for all j     (incremental expansion only)
    (5) SCS_j, FCS_j integer ; DS_ij, U_i >= 0

Cost_DM (cost of demand mismatch) is *not* a function of the above decision
variables -- it measures the accuracy of the demand forecast itself
( sum_i | D_forecast,i - D_true,i | ), fixed once a forecasting model has
been chosen, independent of where stations are subsequently sited. It is
treated as an exogenous planning-risk term, estimated from an out-of-sample
backtest of the forecasting model, and reported alongside Cost_CD and
Cost_IF to reconstruct the study's overall lifecycle cost:
   Cost = a*Cost_CD + b*Cost_DM + c*Cost_IF.

The problem is solved to (near-)global optimality with MILP using the
COIN-OR CBC solver, whose search procedure is a Branch-and-Bound (more
precisely, branch-and-cut) algorithm -- reconciling "solved using MILP"
with "rooted in the Branch-and-Bound methodology": MILP is the
*formulation*, Branch-and-Bound is the *solution algorithm*.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ============================================================================
# SECTION 1: Static geographic reference data (Bengaluru)
# ============================================================================
# Approximate bounding box used for the demand grid (degrees). Roughly spans
# the BBMP administrative area (Yelahanka in the north to Anekal Road /
# Electronics City in the south; Kengeri in the west to Whitefield / KR Puram
# in the east). This -- like the rest of the geographic data below -- is an
# illustrative approximation for cartographic and modelling purposes, not a
# cadastral/survey-grade boundary, consistent with the source study's own
# disclosure that its data are "not extracted from real locations" but are
# designed to "accurately emulate the characteristics of a practical use case."

BBOX = {"lat_min": 12.830, "lat_max": 13.140, "lon_min": 77.450, "lon_max": 77.780}
CITY_CENTER = (12.9716, 77.5946)  # MG Road / Cubbon Park area

BBMP_BOUNDARY_COORDS = [
    [77.5946, 13.145], [77.645, 13.128], [77.700, 13.095], [77.748, 13.045],
    [77.780, 12.995], [77.760, 12.955], [77.740, 12.918], [77.700, 12.878],
    [77.665, 12.845], [77.610, 12.828], [77.560, 12.832], [77.520, 12.850],
    [77.480, 12.888], [77.455, 12.935], [77.462, 12.985], [77.470, 13.030],
    [77.495, 13.075], [77.540, 13.115], [77.5946, 13.145],
]

LOCALITIES = [
    ("MG Road", 12.9757, 77.6067), ("Brigade Road", 12.9716, 77.6089),
    ("Cubbon Park", 12.9763, 77.5928), ("Richmond Town", 12.9634, 77.6034),
    ("Shivaji Nagar", 12.9857, 77.6040), ("Ulsoor", 12.9815, 77.6203),
    ("Indiranagar", 12.9719, 77.6412), ("Domlur", 12.9611, 77.6387),
    ("CV Raman Nagar", 12.9836, 77.6653), ("Koramangala", 12.9352, 77.6146),
    ("BTM Layout", 12.9166, 77.6101), ("HSR Layout", 12.9116, 77.6389),
    ("Bellandur", 12.9257, 77.6784), ("Sarjapur Road", 12.9008, 77.6870),
    ("Marathahalli", 12.9569, 77.7011), ("Whitefield", 12.9698, 77.7500),
    ("ITPL", 12.9863, 77.7370), ("Kadugodi", 12.9930, 77.7650),
    ("Hoodi", 12.9910, 77.7150), ("KR Puram", 13.0033, 77.6970),
    ("Mahadevapura", 12.9906, 77.6868), ("Panathur", 12.9370, 77.7010),
    ("Kundalahalli", 12.9690, 77.7160), ("Varthur", 12.9412, 77.7412),
    ("Electronic City Phase 1", 12.8452, 77.6602),
    ("Electronic City Phase 2", 12.8390, 77.6850),
    ("Bommanahalli", 12.9040, 77.6220), ("Bannerghatta Road", 12.8890, 77.5970),
    ("JP Nagar", 12.9080, 77.5850), ("Jayanagar", 12.9308, 77.5838),
    ("Banashankari", 12.9250, 77.5540), ("Basavanagudi", 12.9420, 77.5730),
    ("Chamrajpet", 12.9560, 77.5670), ("Vijayanagar", 12.9720, 77.5330),
    ("Rajajinagar", 12.9910, 77.5530), ("Malleshwaram", 13.0035, 77.5709),
    ("Yeshwanthpur", 13.0280, 77.5540), ("Peenya", 13.0330, 77.5200),
    ("Nagarbhavi", 12.9600, 77.5070), ("Rajarajeshwari Nagar", 12.9260, 77.5220),
    ("Kengeri", 12.9070, 77.4830), ("Vidyaranyapura", 13.0670, 77.5600),
    ("Yelahanka", 13.1005, 77.5963), ("Yelahanka New Town", 13.1150, 77.5960),
    ("Hebbal", 13.0357, 77.5970), ("RT Nagar", 13.0210, 77.5930),
    ("Sanjaynagar", 13.0180, 77.5760), ("Kammanahalli", 13.0140, 77.6360),
    ("HBR Layout", 13.0230, 77.6350), ("Kalyan Nagar", 13.0230, 77.6440),
    ("Frazer Town", 12.9970, 77.6120), ("Cox Town", 12.9940, 77.6180),
    ("Banaswadi", 13.0140, 77.6510), ("Old Airport Road", 12.9600, 77.6560),
    ("HAL Airport Road", 12.9500, 77.6480), ("Wilson Garden", 12.9540, 77.5940),
    ("Shanti Nagar", 12.9590, 77.5990), ("Attibele", 12.7830, 77.7580),
    ("Anekal", 12.7110, 77.6960), ("Hennur", 13.0430, 77.6280),
    ("Jalahalli", 13.0450, 77.5480),
]

# Base-map tile providers offered in the Spatial Maps tab. OpenStreetMap and
# Esri are free, keyless XYZ tile services whose tiles already render real
# street, neighbourhood and place-name cartography (i.e. a genuine map
# background, not a blank coordinate canvas). The two Google entries use
# Google's raw, unauthenticated `mt1.google.com` tile endpoint so that a
# Google-cartography basemap is available with no setup at all; this is
# fine for local evaluation but is not how Google's terms expect the map to
# be embedded in a deployed app -- see the in-app caption for the
# recommended, key-authenticated alternative (the Maps JavaScript/Static API).
BASEMAP_TILES = {
    "OpenStreetMap (recommended — real street & place names)": {
        "tiles": "OpenStreetMap", "attr": None,
    },
    "Esri World Street Map (real place names, alternate style)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, HERE, Garmin, FAO, NOAA, USGS &mdash; Esri World Street Map",
    },
    "Google Roadmap (unauthenticated demo tiles)": {
        "tiles": "https://mt1.google.com/vt/lyr=m&x={x}&y={y}&z={z}", "attr": "Google Maps",
    },
    "Google Satellite (unauthenticated demo tiles)": {
        "tiles": "https://mt1.google.com/vt/lyr=s&x={x}&y={y}&z={z}", "attr": "Google Maps",
    },
}


def add_place_labels(fmap, df, name_col="name", lat_col="lat", lon_col="lon"):
    """Overlay each row's real place name as a small, permanently-visible
    text label on the map (folium DivIcon) -- so real locality names are
    always visible directly on the map itself, the way a consumer mapping
    app labels places, rather than only appearing on hover."""
    import folium
    for row in df.itertuples():
        name = getattr(row, name_col)
        lat, lon = getattr(row, lat_col), getattr(row, lon_col)
        html = (
            '<div style="font-size:9.5px; font-weight:600; color:#111827; '
            'background:rgba(255,255,255,0.78); padding:0px 3px; border-radius:2px; '
            'white-space:nowrap; transform:translate(-50%, 6px); '
            'box-shadow:0 0 1px rgba(0,0,0,0.4);">'
            f'{name}</div>'
        )
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(icon_size=(1, 1), icon_anchor=(0, 0), html=html),
        ).add_to(fmap)


def bbmp_boundary_geojson():
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "Greater Bengaluru (BBMP) - illustrative boundary",
                "note": "Simplified cartographic approximation, not an official survey boundary.",
            },
            "geometry": {"type": "Polygon", "coordinates": [BBMP_BOUNDARY_COORDS]},
        }],
    }


# ============================================================================
# SECTION 2: Model constants
# ============================================================================
CAP_SCS = 200
CAP_FCS = 400
R_COST_RATIO = 1.5      # r
A_WEIGHT = 1            # a  (Cost_CD weight)
B_WEIGHT = 25           # b  (Cost_DM weight)
C_WEIGHT = 600          # c  (Cost_IF weight)
M_UNMET_PENALTY = 5000  # large penalty per unit of structurally unmet demand

N_DEMAND_SIDE_FULL = 64   # 64 x 64 = 4,096 demand points, per the Dataset section
N_SUPPLY_FULL = 100       # 100 parking locations, per the Dataset section

# Historical demand periods, extended (per the 2026 project brief) from the
# source study's 2010-2018 window through the most recent complete data
# available at the time of writing: sixteen full calendar years (2010-2025,
# x = 0..15) plus one partial-year observation for January-June 2026
# (labelled 2026.5, i.e. x = 16.5 -- the midpoint of the 17th year, matching
# a continuous "years elapsed since 1 Jan 2010" time axis). Demand values at
# every period, including the partial year, are a demand *level* (not a
# prorated annual total), so no additional rescaling is needed for the
# shorter final period.
HISTORY_YEARS = list(range(2010, 2026)) + [2026.5]
FORECAST_TARGET_YEARS = (2027, 2028)  # full calendar years beyond the extended history
EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(h))


def _local_xy(lat, lon, lat0, lon0):
    """Cheap equirectangular projection to km, adequate at city scale."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111.32
    y = (lat - lat0) * 110.57
    return x, y


# ============================================================================
# SECTION 3: Data generation
# ============================================================================

def generate_demand_grid(n_side: int = N_DEMAND_SIDE_FULL) -> pd.DataFrame:
    """n_side x n_side equally spaced demand points across the Bengaluru
    bounding box (paper default: 64x64 = 4,096 points)."""
    lats = np.linspace(BBOX["lat_min"], BBOX["lat_max"], n_side)
    lons = np.linspace(BBOX["lon_min"], BBOX["lon_max"], n_side)
    rows = []
    idx = 0
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            rows.append((idx, r, c, lat, lon))
            idx += 1
    return pd.DataFrame(rows, columns=["point_id", "row", "col", "lat", "lon"])


def _hotspot_weight(lat, lon):
    """Spatial demand-intensity multiplier: higher near the IT corridors
    (Whitefield/ORR/Electronic City) and the core, lower at the fringe."""
    hotspots = [
        (12.9716, 77.5946, 1.00, 6.0), (12.9698, 77.7500, 1.35, 5.0),
        (12.9569, 77.7011, 1.25, 5.0), (12.9257, 77.6784, 1.20, 4.5),
        (12.9116, 77.6389, 1.15, 4.5), (12.8452, 77.6602, 1.30, 5.0),
        (13.0357, 77.5970, 0.95, 4.0), (13.1005, 77.5963, 0.55, 4.0),
        (12.9070, 77.4830, 0.45, 4.0),
    ]
    w = 0.35
    for hlat, hlon, weight, sigma in hotspots:
        d = haversine_km(lat, lon, hlat, hlon)
        w += weight * math.exp(-(d ** 2) / (2 * sigma ** 2))
    return w


def generate_demand_history(grid_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthetic EV charging demand history per demand point, spanning 2010
    through the partial year Jan-Jun 2026 (see HISTORY_YEARS). Growth follows
    a parabolic trend y = a*x^2 + b (x = years elapsed since 1 Jan 2010,
    matching the source study's forecasting section), scaled by a spatial
    hotspot weight and perturbed with point-specific noise so the
    forecasting-model comparison (parabola vs polynomial vs ARIMA) is a
    genuine, data-driven exercise rather than a foregone conclusion."""
    rng = np.random.default_rng(seed)
    n = len(grid_df)
    weights = np.array([_hotspot_weight(r.lat, r.lon) for r in grid_df.itertuples()])
    a_coef = rng.normal(2.2, 0.35, n) * weights
    b_coef = rng.normal(18, 6, n) * weights
    noise_scale = rng.uniform(0.04, 0.09, n)

    records = []
    for year in HISTORY_YEARS:
        x = year - HISTORY_YEARS[0]
        base = a_coef * (x ** 2) + b_coef
        seasonal = 1.0 + 0.03 * math.sin(x)
        noise = rng.normal(1.0, noise_scale, n)
        demand = np.clip(base * seasonal * noise, 0, None)
        for pid, val in zip(grid_df.point_id, demand):
            records.append((pid, year, round(float(val), 2)))
    return pd.DataFrame(records, columns=["point_id", "year", "demand"])


def generate_supply_points(n_points: int = N_SUPPLY_FULL, seed: int = 7) -> pd.DataFrame:
    """Candidate EV-RF (public parking) locations anchored on well-known
    Bengaluru localities (paper default: 100 locations), each with an index,
    coordinates, total parking slots, and existing SCS/FCS counts as of 2018
    -- retained as a fixed baseline inventory throughout this study, even as
    the demand-side dataset is extended through mid-2026, so that the
    "incremental expansion only" constraint (Eq. 7) is anchored to a single,
    unambiguous starting inventory."""
    rng = np.random.default_rng(seed)
    anchors = LOCALITIES
    rows = []
    for idx in range(n_points):
        name, lat0, lon0 = anchors[idx % len(anchors)]
        jitter_lat = rng.normal(0, 0.006) if idx >= len(anchors) else rng.normal(0, 0.0015)
        jitter_lon = rng.normal(0, 0.006) if idx >= len(anchors) else rng.normal(0, 0.0015)
        lat = float(np.clip(lat0 + jitter_lat, BBOX["lat_min"] + 0.002, BBOX["lat_max"] - 0.002))
        lon = float(np.clip(lon0 + jitter_lon, BBOX["lon_min"] + 0.002, BBOX["lon_max"] - 0.002))

        parking_slots = int(rng.integers(8, 55))
        has_infra = rng.random() < 0.35
        if has_infra:
            scs0 = int(rng.integers(0, max(1, parking_slots // 8)))
            fcs0 = int(rng.integers(0, max(1, parking_slots // 16)))
        else:
            scs0, fcs0 = 0, 0
        scs0 = min(scs0, parking_slots)
        fcs0 = min(fcs0, parking_slots - scs0)

        rows.append((idx, f"{name} #{idx}" if idx >= len(anchors) else name,
                     lat, lon, parking_slots, scs0, fcs0))

    return pd.DataFrame(rows, columns=["supply_id", "name", "lat", "lon",
                                        "parking_slots", "scs_2018", "fcs_2018"])


# ============================================================================
# SECTION 4: Demand forecasting
# ============================================================================

def _fit_backtest_metrics(history_df: pd.DataFrame):
    """Aggregate-level (city-wide total demand) backtest comparing candidate
    forecasting models: fit on the first ~75% of observed periods, predict
    the held-out final ~25% (at least 2, at most 4 periods), compare to
    actuals. Generalised to scale with the length of the history series
    (17 periods -- 2010-2025 plus Jan-Jun 2026 -- rather than the source
    study's original 9 annual points), so extending the dataset does not
    silently shrink or distort the holdout window."""
    agg = history_df.groupby("year")["demand"].sum().reset_index().sort_values("year")
    x_all = np.arange(len(agg))
    y_all = agg["demand"].values
    n_test = int(min(4, max(2, round(len(agg) * 0.25))))
    n_train = len(agg) - n_test
    x_train, y_train = x_all[:n_train], y_all[:n_train]
    x_test, y_test = x_all[n_train:], y_all[n_train:]

    def mape(y_true, y_pred):
        return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    results = {}
    A = np.vstack([x_train ** 2, np.ones_like(x_train)]).T
    coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    y_pred = coef[0] * x_test ** 2 + coef[1]
    results["Parabolic (y=ax^2+b)"] = dict(rmse=rmse(y_test, y_pred), mape=mape(y_test, y_pred))

    p3 = np.polyfit(x_train, y_train, deg=3)
    y_pred = np.polyval(p3, x_test)
    results["Polynomial (deg. 3)"] = dict(rmse=rmse(y_test, y_pred), mape=mape(y_test, y_pred))

    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMA(y_train, order=(1, 1, 1)).fit()
            y_pred = m.forecast(steps=len(x_test))
        results["ARIMA(1,1,1)"] = dict(rmse=rmse(y_test, y_pred), mape=mape(y_test, y_pred))
    except Exception as e:
        results["ARIMA(1,1,1)"] = dict(rmse=float("nan"), mape=float("nan"), error=str(e))

    y_pred = np.full_like(y_test, y_train[-1], dtype=float)
    results["Naive (last value)"] = dict(rmse=rmse(y_test, y_pred), mape=mape(y_test, y_pred))

    best = min(results.items(), key=lambda kv: kv[1]["rmse"] if not math.isnan(kv[1]["rmse"]) else 1e18)
    return results, best[0], agg.rename(columns={"demand": "demand"})


def forecast_demand(history_df: pd.DataFrame, target_years=FORECAST_TARGET_YEARS):
    """Per-point parabolic forecast (the model retained for full-resolution
    operational deployment; see docstring / paper Methodology for the
    scalability rationale), plus an exogenous Cost_DM estimate (forecast-error
    risk) derived from in-sample residuals of the fitted parabola."""
    model_comparison, best_model_name, agg_hist = _fit_backtest_metrics(history_df)

    pivot = history_df.pivot(index="point_id", columns="year", values="demand").sort_index()
    x = np.array([y - HISTORY_YEARS[0] for y in pivot.columns])

    forecasts = {}
    resid_sum_sq = []
    for year in target_years:
        x_target = year - HISTORY_YEARS[0]
        A = np.vstack([x ** 2, np.ones_like(x)]).T
        Y = pivot.values.T
        AtA = A.T @ A
        AtY = A.T @ Y
        coefs = np.linalg.solve(AtA, AtY)
        pred = coefs[0] * x_target ** 2 + coefs[1]
        forecasts[year] = np.clip(pred, 0, None)
        fitted = (coefs[0][None, :] * (x[:, None] ** 2)) + coefs[1][None, :]
        resid_sum_sq.append(np.mean((Y - fitted) ** 2, axis=0))

    point_ids = pivot.index.values
    forecast_df = pd.DataFrame({"point_id": point_ids})
    for year in target_years:
        forecast_df[f"demand_{year}"] = forecasts[year]
    forecast_df["forecast_error_std"] = np.sqrt(np.mean(resid_sum_sq, axis=0))

    return forecast_df, model_comparison, best_model_name, agg_hist


# ============================================================================
# SECTION 5: Candidate service pairs (sparsification for tractability)
# ============================================================================

def build_candidate_pairs(demand_df: pd.DataFrame, supply_df: pd.DataFrame,
                           k_nearest: int = 8, max_radius_km: float = 6.0):
    lat0, lon0 = CITY_CENTER
    sx, sy = zip(*[_local_xy(r.lat, r.lon, lat0, lon0) for r in supply_df.itertuples()])
    supply_xy = np.column_stack([sx, sy])
    tree = cKDTree(supply_xy)

    dx, dy = zip(*[_local_xy(r.lat, r.lon, lat0, lon0) for r in demand_df.itertuples()])
    demand_xy = np.column_stack([dx, dy])

    k = min(k_nearest, len(supply_df))
    dists, idxs = tree.query(demand_xy, k=k)
    if k == 1:
        dists, idxs = dists[:, None], idxs[:, None]

    demand_ids = demand_df["point_id"].values
    supply_ids = supply_df["supply_id"].values

    pairs_i, pairs_j, pairs_d = [], [], []
    for row_i in range(len(demand_df)):
        for col in range(k):
            d_km = dists[row_i, col]
            if d_km <= max_radius_km:
                pairs_i.append(demand_ids[row_i])
                pairs_j.append(supply_ids[idxs[row_i, col]])
                pairs_d.append(float(d_km))
    return pd.DataFrame({"point_id": pairs_i, "supply_id": pairs_j, "dist_km": pairs_d})


# ============================================================================
# SECTION 6: Optimisation -- exact MILP (Branch-and-Bound via CBC) + baselines
# ============================================================================

def solve_milp(demand_ids, demand_values, supply_df, candidates_df,
               time_limit_s: int = 120, msg: bool = False):
    import pulp

    t0 = time.time()
    prob = pulp.LpProblem("EV_RF_Siting", pulp.LpMinimize)

    demand_map = dict(zip(demand_ids, demand_values))
    supply_ids = supply_df["supply_id"].tolist()
    ps = dict(zip(supply_df.supply_id, supply_df.parking_slots))
    scs0 = dict(zip(supply_df.supply_id, supply_df.scs_2018))
    fcs0 = dict(zip(supply_df.supply_id, supply_df.fcs_2018))

    SCS = {j: pulp.LpVariable(f"SCS_{j}", lowBound=scs0[j], upBound=ps[j], cat="Integer") for j in supply_ids}
    FCS = {j: pulp.LpVariable(f"FCS_{j}", lowBound=fcs0[j], upBound=ps[j], cat="Integer") for j in supply_ids}

    pairs = list(zip(candidates_df.point_id, candidates_df.supply_id, candidates_df.dist_km))
    DS = {(i, j): pulp.LpVariable(f"DS_{i}_{j}", lowBound=0) for (i, j, d) in pairs}
    U = {i: pulp.LpVariable(f"U_{i}", lowBound=0) for i in demand_ids}

    cost_cd = pulp.lpSum(d * DS[(i, j)] for (i, j, d) in pairs)
    cost_if = pulp.lpSum(SCS[j] + R_COST_RATIO * FCS[j] for j in supply_ids)
    unmet_penalty = pulp.lpSum(U.values())
    prob += A_WEIGHT * cost_cd + C_WEIGHT * cost_if + M_UNMET_PENALTY * unmet_penalty

    for j in supply_ids:
        prob += SCS[j] + FCS[j] <= ps[j], f"slot_limit_{j}"

    by_i, by_j = {}, {}
    for (i, j, d) in pairs:
        by_i.setdefault(i, []).append(j)
        by_j.setdefault(j, []).append(i)

    for i in demand_ids:
        js = by_i.get(i, [])
        prob += pulp.lpSum(DS[(i, j)] for j in js) + U[i] == demand_map[i], f"balance_{i}"
    for j in supply_ids:
        is_ = by_j.get(j, [])
        prob += (pulp.lpSum(DS[(i, j)] for i in is_)
                 <= CAP_SCS * SCS[j] + CAP_FCS * FCS[j]), f"capacity_{j}"

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s)
    status = prob.solve(solver)
    solve_time = time.time() - t0

    scs_sol = {j: int(round(SCS[j].value())) for j in supply_ids}
    fcs_sol = {j: int(round(FCS[j].value())) for j in supply_ids}
    ds_sol = {(i, j): DS[(i, j)].value() or 0.0 for (i, j, d) in pairs}
    u_sol = {i: U[i].value() or 0.0 for i in demand_ids}

    return dict(
        method="MILP (Branch-and-Bound, CBC)", status=pulp.LpStatus[status],
        solve_time_s=solve_time, objective=pulp.value(prob.objective),
        cost_cd=sum(d * ds_sol[(i, j)] for (i, j, d) in pairs),
        cost_if=sum(scs_sol[j] + R_COST_RATIO * fcs_sol[j] for j in supply_ids),
        unmet_total=sum(u_sol.values()),
        scs=scs_sol, fcs=fcs_sol, ds=ds_sol, u=u_sol,
    )


def solve_greedy(demand_ids, demand_values, supply_df, candidates_df):
    """Baseline 1: nearest-facility greedy heuristic (no re-optimisation)."""
    t0 = time.time()
    demand_map = dict(zip(demand_ids, demand_values))
    ps = dict(zip(supply_df.supply_id, supply_df.parking_slots))
    scs = dict(zip(supply_df.supply_id, supply_df.scs_2018))
    fcs = dict(zip(supply_df.supply_id, supply_df.fcs_2018))
    remaining_cap = {j: CAP_SCS * scs[j] + CAP_FCS * fcs[j] for j in ps}
    remaining_slots = {j: ps[j] - scs[j] - fcs[j] for j in ps}

    by_i = {}
    for r in candidates_df.itertuples():
        by_i.setdefault(r.point_id, []).append((r.dist_km, r.supply_id))
    for i in by_i:
        by_i[i].sort()

    ds, u = {}, {i: 0.0 for i in demand_ids}
    for i in demand_ids:
        need = demand_map[i]
        for dist, j in by_i.get(i, []):
            if need <= 1e-9:
                break
            if remaining_cap[j] < need and remaining_slots[j] > 0:
                add_scs = min(remaining_slots[j], math.ceil((need - remaining_cap[j]) / CAP_SCS))
                scs[j] += add_scs
                remaining_slots[j] -= add_scs
                remaining_cap[j] += add_scs * CAP_SCS
            serve = min(need, remaining_cap[j])
            if serve > 0:
                ds[(i, j)] = ds.get((i, j), 0.0) + serve
                remaining_cap[j] -= serve
                need -= serve
        u[i] = max(need, 0.0)

    cost_cd = sum(d * ds.get((r.point_id, r.supply_id), 0.0)
                  for r, d in zip(candidates_df.itertuples(), candidates_df.dist_km))
    cost_if = sum(scs[j] + R_COST_RATIO * fcs[j] for j in ps)
    unmet_total = sum(u.values())
    objective = A_WEIGHT * cost_cd + C_WEIGHT * cost_if + M_UNMET_PENALTY * unmet_total

    return dict(method="Greedy nearest-assignment (heuristic)", status="Heuristic",
                solve_time_s=time.time() - t0, objective=objective,
                cost_cd=cost_cd, cost_if=cost_if, unmet_total=unmet_total,
                scs=scs, fcs=fcs, ds=ds, u=u)


def solve_lp_relaxation(demand_ids, demand_values, supply_df, candidates_df,
                         time_limit_s: int = 60, msg: bool = False):
    """Baseline 2: relax SCS_j / FCS_j to continuous, solve the LP, round up
    to the nearest feasible integer, then re-derive a feasible assignment."""
    import pulp

    t0 = time.time()
    prob = pulp.LpProblem("EV_RF_Siting_LP", pulp.LpMinimize)
    demand_map = dict(zip(demand_ids, demand_values))
    supply_ids = supply_df["supply_id"].tolist()
    ps = dict(zip(supply_df.supply_id, supply_df.parking_slots))
    scs0 = dict(zip(supply_df.supply_id, supply_df.scs_2018))
    fcs0 = dict(zip(supply_df.supply_id, supply_df.fcs_2018))

    SCS = {j: pulp.LpVariable(f"SCSr_{j}", lowBound=scs0[j], upBound=ps[j]) for j in supply_ids}
    FCS = {j: pulp.LpVariable(f"FCSr_{j}", lowBound=fcs0[j], upBound=ps[j]) for j in supply_ids}
    pairs = list(zip(candidates_df.point_id, candidates_df.supply_id, candidates_df.dist_km))
    DS = {(i, j): pulp.LpVariable(f"DSr_{i}_{j}", lowBound=0) for (i, j, d) in pairs}
    U = {i: pulp.LpVariable(f"Ur_{i}", lowBound=0) for i in demand_ids}

    cost_cd = pulp.lpSum(d * DS[(i, j)] for (i, j, d) in pairs)
    cost_if = pulp.lpSum(SCS[j] + R_COST_RATIO * FCS[j] for j in supply_ids)
    unmet_penalty = pulp.lpSum(U.values())
    prob += A_WEIGHT * cost_cd + C_WEIGHT * cost_if + M_UNMET_PENALTY * unmet_penalty

    for j in supply_ids:
        prob += SCS[j] + FCS[j] <= ps[j]
    by_i, by_j = {}, {}
    for (i, j, d) in pairs:
        by_i.setdefault(i, []).append(j)
        by_j.setdefault(j, []).append(i)
    for i in demand_ids:
        js = by_i.get(i, [])
        prob += pulp.lpSum(DS[(i, j)] for j in js) + U[i] == demand_map[i]
    for j in supply_ids:
        is_ = by_j.get(j, [])
        prob += pulp.lpSum(DS[(i, j)] for i in is_) <= CAP_SCS * SCS[j] + CAP_FCS * FCS[j]

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s)
    prob.solve(solver)

    scs_r = {j: SCS[j].value() or 0.0 for j in supply_ids}
    fcs_r = {j: FCS[j].value() or 0.0 for j in supply_ids}
    scs_i, fcs_i = {}, {}
    for j in supply_ids:
        s = max(scs0[j], math.ceil(scs_r[j] - 1e-6))
        f = max(fcs0[j], math.ceil(fcs_r[j] - 1e-6))
        if s + f > ps[j]:
            f = max(fcs0[j], ps[j] - s)
        scs_i[j], fcs_i[j] = s, f

    remaining_cap = {j: CAP_SCS * scs_i[j] + CAP_FCS * fcs_i[j] for j in supply_ids}
    by_i_sorted = {}
    for r in candidates_df.itertuples():
        by_i_sorted.setdefault(r.point_id, []).append((r.dist_km, r.supply_id))
    for i in by_i_sorted:
        by_i_sorted[i].sort()

    ds_final, u_final = {}, {i: 0.0 for i in demand_ids}
    for i in demand_ids:
        need = demand_map[i]
        for dist, j in by_i_sorted.get(i, []):
            if need <= 1e-9:
                break
            serve = min(need, remaining_cap[j])
            if serve > 0:
                ds_final[(i, j)] = ds_final.get((i, j), 0.0) + serve
                remaining_cap[j] -= serve
                need -= serve
        u_final[i] = max(need, 0.0)

    cost_cd = sum(d * ds_final.get((i, j), 0.0) for (i, j, d) in pairs)
    cost_if = sum(scs_i[j] + R_COST_RATIO * fcs_i[j] for j in supply_ids)
    unmet_total = sum(u_final.values())
    objective = A_WEIGHT * cost_cd + C_WEIGHT * cost_if + M_UNMET_PENALTY * unmet_total

    return dict(method="LP-relaxation + rounding (heuristic)", status="Heuristic",
                solve_time_s=time.time() - t0, objective=objective,
                cost_cd=cost_cd, cost_if=cost_if, unmet_total=unmet_total,
                scs=scs_i, fcs=fcs_i, ds=ds_final, u=u_final)


# ============================================================================
# SECTION 7: KPI computation
# ============================================================================

def compute_kpis(result: dict, demand_ids, demand_values, supply_df,
                  forecast_error_std=None, b_weight=B_WEIGHT):
    demand_total = float(np.sum(demand_values))
    served_total = demand_total - result["unmet_total"]
    coverage_pct = 100.0 * served_total / demand_total if demand_total > 0 else 0.0
    n_served_pairs = sum(1 for v in result["ds"].values() if v > 1e-6)
    avg_dist = (result["cost_cd"] / served_total) if served_total > 0 else 0.0

    scs0_total = int(supply_df.scs_2018.sum())
    fcs0_total = int(supply_df.fcs_2018.sum())
    scs_total = int(sum(result["scs"].values()))
    fcs_total = int(sum(result["fcs"].values()))
    new_scs = scs_total - scs0_total
    new_fcs = fcs_total - fcs0_total

    unit_cost_scs_lakh = 1.5
    unit_cost_fcs_lakh = unit_cost_scs_lakh * R_COST_RATIO
    capex_lakh = new_scs * unit_cost_scs_lakh + new_fcs * unit_cost_fcs_lakh

    cost_dm = float(np.sum(forecast_error_std)) if forecast_error_std is not None else None
    total_lifecycle_cost = A_WEIGHT * result["cost_cd"] + C_WEIGHT * result["cost_if"]
    if cost_dm is not None:
        total_lifecycle_cost += b_weight * cost_dm
    cost_per_unit_served = (total_lifecycle_cost / served_total) if served_total > 0 else float("nan")

    return dict(
        demand_total=demand_total, served_total=served_total, coverage_pct=coverage_pct,
        unmet_total=result["unmet_total"], avg_travel_distance_km=avg_dist,
        n_active_assignments=n_served_pairs,
        scs_2018=scs0_total, fcs_2018=fcs0_total, scs_optimal=scs_total, fcs_optimal=fcs_total,
        new_scs=new_scs, new_fcs=new_fcs, capex_est_inr_lakh=capex_lakh,
        cost_cd=result["cost_cd"], cost_if=result["cost_if"], cost_dm=cost_dm,
        objective=result["objective"], total_lifecycle_cost=total_lifecycle_cost,
        cost_per_unit_served=cost_per_unit_served,
        solve_time_s=result["solve_time_s"], status=result["status"], method=result["method"],
    )


# ============================================================================
# SECTION 8: GeoJSON export
# ============================================================================

def demand_points_geojson(demand_df, value_series=None, value_name="demand"):
    features = []
    values = value_series if value_series is not None else [None] * len(demand_df)
    for row, val in zip(demand_df.itertuples(), values):
        props = {"point_id": int(row.point_id), "row": int(row.row), "col": int(row.col)}
        if val is not None:
            props[value_name] = float(val)
        features.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
                          "properties": props})
    return {"type": "FeatureCollection", "features": features}


def supply_points_geojson(supply_df, scs_col="scs_2018", fcs_col="fcs_2018", extra_cols=None):
    features = []
    extra_cols = extra_cols or []
    for row in supply_df.itertuples():
        props = {"supply_id": int(row.supply_id), "name": row.name,
                  "parking_slots": int(row.parking_slots),
                  "scs": int(getattr(row, scs_col)), "fcs": int(getattr(row, fcs_col))}
        for c in extra_cols:
            v = getattr(row, c, None)
            if v is not None:
                props[c] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
        features.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
                          "properties": props})
    return {"type": "FeatureCollection", "features": features}


# ============================================================================
# SECTION 9: Streamlit UI
# ============================================================================

def _run_streamlit_app():
    global A_WEIGHT, B_WEIGHT, C_WEIGHT, R_COST_RATIO
    import streamlit as st
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import HeatMap

    st.set_page_config(page_title="EV-RF Network Optimiser",
                        page_icon="\U0001F50C", layout="wide")

    st.markdown("""
        <style>
        .metric-card {background:#0f172a10;border-radius:10px;padding:0.6rem 1rem;border:1px solid #e5e7eb;}
        .block-container {padding-top:1.6rem;}
        h1, h2, h3 {font-family: "Source Sans Pro", sans-serif;}
        </style>
    """, unsafe_allow_html=True)

    st.title("\U0001F50C EV-RF Network Optimiser")
    st.caption("Optimal siting and capacity planning of Electric Vehicle Recharging Facilities "
               "(EV-RF) — an exact Mixed-Integer Linear Programming (Branch-and-Bound) decision-support tool")

    # ---------------- Session state ----------------
    ss = st.session_state
    ss.setdefault("demand_df", None)
    ss.setdefault("supply_df", None)
    ss.setdefault("hist_df", None)
    ss.setdefault("forecast_df", None)
    ss.setdefault("model_comparison", None)
    ss.setdefault("best_model_name", None)
    ss.setdefault("agg_hist", None)
    ss.setdefault("candidates_df", None)
    ss.setdefault("results", {})   # method -> result dict
    ss.setdefault("kpis", {})      # method -> kpi dict

    # ---------------- Sidebar controls ----------------
    with st.sidebar:
        st.header("Study configuration")
        st.markdown("**Study area:** Bengaluru (Bengaluru Urban), Karnataka, India")

        n_side = st.select_slider("Demand-grid resolution (n x n)",
                                   options=[16, 24, 32, 48, 64], value=24,
                                   help="Paper scale is 64x64 = 4,096 demand points. "
                                        "Use a smaller grid for fast interactive exploration; "
                                        "use 64 for the full paper-scale run (slower).")
        st.caption("Note: at coarser grids, 100 supply points can be enough capacity for every "
                   "heuristic to reach similar coverage, which narrows the MILP-vs-heuristic gap. "
                   "The 64x64 paper-scale run is the regime where the exact Branch-and-Bound "
                   "solution's advantage over the heuristics is most pronounced (see the paper).")
        n_supply = st.slider("Number of EV-RF supply (parking) locations", 20, 100, 100, step=5)

        st.markdown("---")
        st.subheader("Service-reach assumptions")
        k_nearest = st.slider("Candidate supply points per demand point (k-nearest)", 4, 15, 8)
        max_radius = st.slider("Maximum practical detour radius R_max (km)", 2.0, 10.0, 6.0, step=0.5)

        st.markdown("---")
        st.subheader("Cost parameters (source study defaults)")
        a_w = st.number_input("a (Cost_CD weight)", value=float(A_WEIGHT), step=1.0)
        b_w = st.number_input("b (Cost_DM weight)", value=float(B_WEIGHT), step=1.0)
        c_w = st.number_input("c (Cost_IF weight)", value=float(C_WEIGHT), step=10.0)
        r_ratio = st.number_input("r (FCS/SCS cost ratio)", value=float(R_COST_RATIO), step=0.1)

        st.markdown("---")
        forecast_year = st.selectbox("Forecast target year", list(FORECAST_TARGET_YEARS), index=0)
        time_limit = st.slider("MILP time limit (seconds)", 10, 300, 60, step=10,
                                help="CBC Branch-and-Bound wall-clock budget. The full 64x64 "
                                     "paper-scale run needs ~150-180s to reach a ~0% optimality gap.")
        seed = st.number_input("Random seed (synthetic data)", value=42, step=1)

        st.markdown("---")
        gen_clicked = st.button("1) Upload Data", width='stretch')
        fc_clicked = st.button("2) Run demand forecasting", width='stretch')
        opt_clicked = st.button("3) Run optimisation (MILP + baselines)", width='stretch')

    # apply live parameter overrides
    A_WEIGHT, B_WEIGHT, C_WEIGHT, R_COST_RATIO = a_w, b_w, c_w, r_ratio

    # ---------------- Actions ----------------
    if gen_clicked:
        with st.spinner("Generating geo-referenced demand grid, demand history and supply network..."):
            ss.demand_df = generate_demand_grid(n_side)
            ss.supply_df = generate_supply_points(n_supply, seed=int(seed) + 7)
            ss.hist_df = generate_demand_history(ss.demand_df, seed=int(seed))
            ss.forecast_df = ss.model_comparison = ss.best_model_name = ss.agg_hist = None
            ss.candidates_df = None
            ss.results, ss.kpis = {}, {}
        st.success(f"Uploaded {len(ss.demand_df)} demand points and {len(ss.supply_df)} "
                   f"candidate EV-RF supply locations across Bengaluru.")

    if fc_clicked:
        if ss.hist_df is None:
            st.warning("Upload Data First (step 1).")
        else:
            with st.spinner("Backtesting forecasting models and fitting per-point demand forecasts..."):
                (ss.forecast_df, ss.model_comparison,
                 ss.best_model_name, ss.agg_hist) = forecast_demand(
                    ss.hist_df, target_years=FORECAST_TARGET_YEARS)
            st.success(f"Forecasting complete. Best-fit model (out-of-sample backtest): {ss.best_model_name}")

    if opt_clicked:
        if ss.forecast_df is None:
            st.warning("Run demand forecasting first (step 2).")
        else:
            with st.spinner("Building candidate service pairs..."):
                ss.candidates_df = build_candidate_pairs(ss.demand_df, ss.supply_df,
                                                          k_nearest=k_nearest, max_radius_km=max_radius)
            demand_ids = ss.forecast_df.point_id.values
            demand_values = ss.forecast_df[f"demand_{forecast_year}"].values
            fe_std = ss.forecast_df.forecast_error_std.values

            with st.spinner(f"Solving exact MILP with Branch-and-Bound (CBC, up to {time_limit}s)..."):
                res_milp = solve_milp(demand_ids, demand_values, ss.supply_df, ss.candidates_df,
                                       time_limit_s=int(time_limit))
            with st.spinner("Running heuristic baselines for comparative analysis..."):
                res_greedy = solve_greedy(demand_ids, demand_values, ss.supply_df, ss.candidates_df)
                res_lp = solve_lp_relaxation(demand_ids, demand_values, ss.supply_df, ss.candidates_df,
                                              time_limit_s=min(60, int(time_limit)))

            ss.results = {"MILP": res_milp, "LP-relaxation": res_lp, "Greedy": res_greedy}
            ss.kpis = {k: compute_kpis(v, demand_ids, demand_values, ss.supply_df, fe_std, b_w)
                       for k, v in ss.results.items()}
            st.success(f"Optimisation complete. MILP status: {res_milp['status']} "
                       f"in {res_milp['solve_time_s']:.1f}s.")

    # ---------------- Tabs ----------------
    tabs = st.tabs(["Overview & Model", "Data", "Forecasting", "Optimisation & KPIs",
                    "Spatial Maps (GeoJSON)", "Comparative Analysis", "Downloads"])

    # ---- Overview & Model ----
    with tabs[0]:
        st.subheader("Problem framing")
        st.markdown("""
This tool locates and sizes Electric Vehicle Recharging Facilities (EV-RF) across Bengaluru.
The city is discretised into a regular grid of demand points (block centroids); a curated set of
candidate parking locations act as supply points where slow-charging stations (SCS) and
fast-charging stations (FCS) can be installed. The model decides **how many SCS/FCS to add at
each candidate site** and **which demand blocks each site serves**, so as to minimise total system
cost while expanding only incrementally beyond the existing 2018 baseline infrastructure. Demand
history now spans 2010 through the partial year Jan-Jun 2026, with per-point forecasts produced
for full-year 2027 and 2028.
        """)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Objective**")
            st.latex(r"""
\min \; a\!\!\sum_{(i,j)\in A}\! \text{Dist}_{ij}\,DS_{ij} \;+\; c\sum_{j} \big(SCS_j + r\,FCS_j\big)
\;+\; M\sum_i U_i
            """)
            st.markdown("**Demand balance**")
            st.latex(r"\sum_{j:(i,j)\in A} DS_{ij} + U_i = D_i \quad \forall i")
            st.markdown("**Capacity**")
            st.latex(r"\sum_{i:(i,j)\in A} DS_{ij} \le \text{Cap}_{SCS}\,SCS_j + \text{Cap}_{FCS}\,FCS_j \quad \forall j")
        with c2:
            st.markdown("**Parking-slot limit**")
            st.latex(r"SCS_j + FCS_j \le PS_j \quad \forall j")
            st.markdown("**Incremental expansion**")
            st.latex(r"SCS_j \ge SCS_j^{2018}, \quad FCS_j \ge FCS_j^{2018}")
            st.markdown("**Integrality**")
            st.latex(r"SCS_j, FCS_j \in \mathbb{Z}_{\ge 0}; \quad DS_{ij}, U_i \ge 0")

        st.info(
            "**Solved via:** Mixed-Integer Linear Programming (the *formulation*), using the "
            "COIN-OR CBC solver, whose search procedure is a **Branch-and-Bound / branch-and-cut "
            "algorithm** (the *solution method*). `Cost_DM` (demand-forecast-mismatch risk) is "
            "exogenous to the siting decision and is reported as a lifecycle-cost KPI rather than "
            "optimised over.")

        st.markdown("Use the sidebar to (1) Upload Data, (2) run forecasting, "
                     "(3) run the optimisation, then explore the tabs to the right.")

    # ---- Data ----
    with tabs[1]:
        st.subheader("Uploaded datasets")
        if ss.demand_df is None:
            st.warning("Click **1) Upload Data** in the sidebar to begin.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Demand grid** ({len(ss.demand_df)} points)")
                st.dataframe(ss.demand_df.head(10), width='stretch', height=220)
                st.download_button("Download Demand_Grid.csv", ss.demand_df.to_csv(index=False),
                                    "Demand_Grid.csv", "text/csv")
                st.markdown(f"**Demand history 2010 – H1 2026** ({len(ss.hist_df)} rows, long format)")
                st.dataframe(ss.hist_df.head(10), width='stretch', height=220)
                st.download_button("Download Demand_History.csv", ss.hist_df.to_csv(index=False),
                                    "Demand_History.csv", "text/csv")
            with c2:
                st.markdown(f"**Existing EV infrastructure, 2018** ({len(ss.supply_df)} locations)")
                st.dataframe(ss.supply_df, width='stretch', height=460)
                st.download_button("Download Existing_EV_infrastructure_2018.csv",
                                    ss.supply_df.to_csv(index=False),
                                    "Existing_EV_infrastructure_2018.csv", "text/csv")

    # ---- Forecasting ----
    with tabs[2]:
        st.subheader("Demand forecasting: model comparison and selection")
        if ss.forecast_df is None:
            st.warning("Run demand forecasting in the sidebar (step 2).")
        else:
            comp_df = pd.DataFrame(ss.model_comparison).T.reset_index().rename(columns={"index": "Model"})
            st.dataframe(comp_df, width='stretch')
            st.markdown(f"**Best out-of-sample fit (held-out final periods backtest):** `{ss.best_model_name}`")
            st.caption("Operational per-point forecasting uses the closed-form Parabolic model "
                       "(y = ax²+b) for all 4,096 points: it is vectorisable across the full "
                       "grid in milliseconds, whereas fitting one ARIMA model per demand point does "
                       "not scale as cheaply, even where ARIMA is occasionally more accurate at the "
                       "aggregate level. This trade-off is discussed in the accompanying paper.")

            st.line_chart(ss.agg_hist.set_index("year")["demand"], height=280)

            fc_year_cols = [c for c in ss.forecast_df.columns if c.startswith("demand_")]
            metric_cols = st.columns(len(fc_year_cols) + 1)
            for col, fc_col in zip(metric_cols, fc_year_cols):
                yr_label = fc_col.replace("demand_", "")
                col.metric(f"Forecast total demand, {yr_label}", f"{ss.forecast_df[fc_col].sum():,.0f}")
            c3 = metric_cols[-1]
            c3.metric("Mean forecast-error std (Cost_DM input)",
                      f"{ss.forecast_df.forecast_error_std.mean():.2f}")
            st.download_button("Download Demand_Forecast.csv", ss.forecast_df.to_csv(index=False),
                                "Demand_Forecast.csv", "text/csv")

    # ---- Optimisation & KPIs ----
    with tabs[3]:
        st.subheader("Optimised EV-RF network (exact MILP / Branch-and-Bound)")
        if not ss.results:
            st.warning("Run the optimisation in the sidebar (step 3).")
        else:
            kpi = ss.kpis["MILP"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Demand coverage", f"{kpi['coverage_pct']:.1f}%")
            m2.metric("Avg. travel distance", f"{kpi['avg_travel_distance_km']:.2f} km")
            m3.metric("New SCS installed", f"{kpi['new_scs']:,}")
            m4.metric("New FCS installed", f"{kpi['new_fcs']:,}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Cost of Customer Dissatisfaction", f"{kpi['cost_cd']:,.0f}")
            m6.metric("Cost of Infrastructure", f"{kpi['cost_if']:,.0f}")
            m7.metric("Cost of Demand Mismatch", f"{kpi['cost_dm']:.1f}")
            m8.metric("Est. capex (INR lakh)", f"{kpi['capex_est_inr_lakh']:,.1f}")

            m9, m10, m11 = st.columns(3)
            m9.metric("Solver status", kpi["status"])
            m10.metric("Solve time", f"{kpi['solve_time_s']:.1f} s")
            m11.metric("Cost per unit demand served", f"{kpi['cost_per_unit_served']:.3f}")

            st.markdown("**Station-level optimisation results**")
            res = ss.results["MILP"]
            out = ss.supply_df.copy()
            out["scs_optimal"] = out.supply_id.map(res["scs"])
            out["fcs_optimal"] = out.supply_id.map(res["fcs"])
            out["new_scs"] = out.scs_optimal - out.scs_2018
            out["new_fcs"] = out.fcs_optimal - out.fcs_2018
            st.dataframe(out, width='stretch', height=350)
            st.download_button("Download Optimised_EV_Infrastructure.csv", out.to_csv(index=False),
                                "Optimised_EV_Infrastructure.csv", "text/csv")

    # ---- Spatial Maps ----
    with tabs[4]:
        st.subheader("Interactive spatial maps (built from GeoJSON)")
        if ss.demand_df is None:
            st.warning("Upload Data First (step 1).")
        else:
            map_choice = st.radio("Layer", ["Forecast demand heatmap", "Existing (2018) infrastructure",
                                             "Optimised infrastructure", "Coverage (served vs unmet)"],
                                   horizontal=True)

            mc1, mc2 = st.columns([2, 1])
            with mc1:
                tile_choice = st.selectbox("Base map style", list(BASEMAP_TILES.keys()), index=0,
                    help="OpenStreetMap and Esri are free, keyless basemaps whose tiles already carry "
                         "real street, neighbourhood and place-name labels (i.e. a genuine map, not a "
                         "blank coordinate canvas). The Google options render Google's own cartography "
                         "using Google's raw, unauthenticated tile endpoint for quick visual comparison; "
                         "Google's terms require a Google Maps Platform API key and the official Maps "
                         "JavaScript/Static API for any production or publicly shared deployment, so "
                         "swap in your own key (see the note below) before relying on them beyond "
                         "local evaluation.")
            with mc2:
                show_place_labels = st.checkbox("Show place-name labels", value=True,
                    help="Overlay each EV-RF site's real locality name permanently on the map, rather "
                         "than only on hover.")
            if tile_choice.startswith("Google"):
                st.caption(
                    "⚠️ Using Google's raw map tiles without an API key -- fine for local, "
                    "one-off evaluation, but not licensed for a deployed or publicly shared app. "
                    "For production use, request a free-tier Google Maps Platform API key at "
                    "https://console.cloud.google.com/google/maps-apis and swap this layer for the "
                    "official `googlemaps`/Maps JavaScript API integration.")

            tile_spec = BASEMAP_TILES[tile_choice]
            fmap = folium.Map(location=[CITY_CENTER[0], CITY_CENTER[1]], zoom_start=11,
                               tiles=tile_spec["tiles"], attr=tile_spec["attr"])
            folium.GeoJson(bbmp_boundary_geojson(), name="BBMP boundary (illustrative)",
                            style_function=lambda x: {"color": "#374151", "weight": 2, "fillOpacity": 0}
                            ).add_to(fmap)

            if map_choice == "Forecast demand heatmap":
                if ss.forecast_df is not None:
                    merged = ss.demand_df.merge(ss.forecast_df, on="point_id")
                    heat_data = merged[["lat", "lon", f"demand_{forecast_year}"]].values.tolist()
                    HeatMap(heat_data, radius=14, blur=18, max_zoom=13).add_to(fmap)
                else:
                    geo = demand_points_geojson(ss.demand_df)
                    for feat in geo["features"][::max(1, len(geo['features']) // 2000)]:
                        lon, lat = feat["geometry"]["coordinates"]
                        folium.CircleMarker([lat, lon], radius=1.5, color="#f59e0b").add_to(fmap)
                    st.caption("Run forecasting for a demand-weighted heatmap; showing raw grid for now.")
                if show_place_labels:
                    add_place_labels(fmap, ss.supply_df)

            elif map_choice == "Existing (2018) infrastructure":
                geo = supply_points_geojson(ss.supply_df)
                folium.GeoJson(
                    geo, name="Existing EV-RF (2018)",
                    marker=folium.CircleMarker(radius=5),
                    style_function=lambda f: {
                        "fillColor": "#2563eb" if (f["properties"]["scs"] + f["properties"]["fcs"]) > 0 else "#9ca3af",
                        "color": "#111827", "weight": 0.6, "fillOpacity": 0.85},
                    tooltip=folium.GeoJsonTooltip(fields=["name", "parking_slots", "scs", "fcs"]),
                ).add_to(fmap)
                if show_place_labels:
                    add_place_labels(fmap, ss.supply_df)

            elif map_choice == "Optimised infrastructure":
                if "MILP" in ss.results:
                    res = ss.results["MILP"]
                    out = ss.supply_df.copy()
                    out["scs_opt"] = out.supply_id.map(res["scs"])
                    out["fcs_opt"] = out.supply_id.map(res["fcs"])
                    geo = supply_points_geojson(out, "scs_opt", "fcs_opt")
                    folium.GeoJson(
                        geo, name="Optimised EV-RF",
                        style_function=lambda f: {
                            "fillColor": "#16a34a", "color": "#111827", "weight": 0.6, "fillOpacity": 0.85},
                        tooltip=folium.GeoJsonTooltip(fields=["name", "parking_slots", "scs", "fcs"]),
                    ).add_to(fmap)
                    if show_place_labels:
                        add_place_labels(fmap, out)
                else:
                    st.warning("Run optimisation first (step 3).")

            else:  # Coverage
                if "MILP" in ss.results:
                    res = ss.results["MILP"]
                    demand_ids = ss.forecast_df.point_id.values
                    demand_values = ss.forecast_df[f"demand_{forecast_year}"].values
                    served = np.zeros(len(demand_ids))
                    id_to_idx = {pid: i for i, pid in enumerate(demand_ids)}
                    for (i, j), v in res["ds"].items():
                        served[id_to_idx[i]] += v
                    served_frac = np.clip(served / np.maximum(demand_values, 1e-9), 0, 1)
                    merged = ss.demand_df.copy()
                    merged["served_frac"] = served_frac
                    for row in merged.itertuples():
                        color = "#16a34a" if row.served_frac > 0.99 else ("#f59e0b" if row.served_frac > 0.4 else "#dc2626")
                        folium.CircleMarker([row.lat, row.lon], radius=2, color=color, fill=True,
                                             fill_opacity=0.7, weight=0).add_to(fmap)
                    if show_place_labels:
                        add_place_labels(fmap, ss.supply_df)
                else:
                    st.warning("Run optimisation first (step 3).")

            st_folium(fmap, height=560, width='stretch', returned_objects=[])

            st.markdown("**Raw GeoJSON downloads**")
            gjc1, gjc2, gjc3 = st.columns(3)
            gjc1.download_button("Boundary.geojson", json.dumps(bbmp_boundary_geojson()),
                                  "Bengaluru_Boundary.geojson", "application/geo+json")
            demand_geo = demand_points_geojson(
                ss.demand_df, ss.forecast_df[f"demand_{forecast_year}"].values if ss.forecast_df is not None else None,
                f"demand_{forecast_year}")
            gjc2.download_button("Demand_points.geojson", json.dumps(demand_geo),
                                  "Demand_Points.geojson", "application/geo+json")
            supply_geo = supply_points_geojson(ss.supply_df)
            gjc3.download_button("Supply_points.geojson", json.dumps(supply_geo),
                                  "Supply_Points.geojson", "application/geo+json")

    # ---- Comparative Analysis ----
    with tabs[5]:
        st.subheader("Comparative analysis: Branch-and-Bound MILP vs. heuristic baselines")
        if not ss.kpis:
            st.warning("Run the optimisation in the sidebar (step 3).")
        else:
            rows = []
            for name, kpi in ss.kpis.items():
                rows.append({
                    "Method": kpi["method"], "Coverage (%)": round(kpi["coverage_pct"], 2),
                    "Avg. distance (km)": round(kpi["avg_travel_distance_km"], 3),
                    "Cost_CD": round(kpi["cost_cd"], 1), "Cost_IF": round(kpi["cost_if"], 1),
                    "Cost / unit served": round(kpi["cost_per_unit_served"], 3),
                    "Solve time (s)": round(kpi["solve_time_s"], 2),
                })
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df, width='stretch')

            cc1, cc2, cc3 = st.columns(3)
            cc1.bar_chart(cmp_df.set_index("Method")["Coverage (%)"])
            cc2.bar_chart(cmp_df.set_index("Method")["Cost / unit served"])
            cc3.bar_chart(cmp_df.set_index("Method")["Solve time (s)"])
            st.caption("Coverage and cost-efficiency reward the exact Branch-and-Bound MILP solution; "
                       "the heuristics trade solution quality for (near-)instant computation — a "
                       "classic optimality-vs-speed trade-off, quantified here rather than asserted.")
            st.download_button("Download Comparative_Analysis.csv", cmp_df.to_csv(index=False),
                                "Comparative_Analysis.csv", "text/csv")

    # ---- Downloads (bundle) ----
    with tabs[6]:
        st.subheader("Download everything as one archive")
        if ss.demand_df is None:
            st.warning("Upload Data First (step 1).")
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Demand_Grid.csv", ss.demand_df.to_csv(index=False))
                zf.writestr("Demand_History.csv", ss.hist_df.to_csv(index=False))
                zf.writestr("Existing_EV_infrastructure_2018.csv", ss.supply_df.to_csv(index=False))
                zf.writestr("Boundary.geojson", json.dumps(bbmp_boundary_geojson()))
                zf.writestr("Supply_Points.geojson", json.dumps(supply_points_geojson(ss.supply_df)))
                if ss.forecast_df is not None:
                    zf.writestr("Demand_Forecast.csv", ss.forecast_df.to_csv(index=False))
                    zf.writestr("Demand_Points.geojson", json.dumps(demand_points_geojson(
                        ss.demand_df, ss.forecast_df[f"demand_{forecast_year}"].values, f"demand_{forecast_year}")))
                if ss.results:
                    res = ss.results["MILP"]
                    out = ss.supply_df.copy()
                    out["scs_optimal"] = out.supply_id.map(res["scs"])
                    out["fcs_optimal"] = out.supply_id.map(res["fcs"])
                    zf.writestr("Optimised_EV_Infrastructure.csv", out.to_csv(index=False))
                    rows = [{"Method": k["method"], **{kk: vv for kk, vv in k.items() if kk not in ("method",)}}
                            for k in ss.kpis.values()]
                    zf.writestr("Comparative_Analysis.csv", pd.DataFrame(rows).to_csv(index=False))
            st.download_button("\U0001F4E6 Download all data & results (.zip)", buf.getvalue(),
                                "Bengaluru_EV_RF_data_and_results.zip", "application/zip",
                                width='stretch')


if __name__ == "__main__":
    # This file is a Streamlit app: it is designed to be launched with
    #     streamlit run app.py
    # rather than executed as a plain Python script. PyCharm's (and most
    # IDEs') default "Run" button, however, invokes `python app.py`
    # directly, which does not start the Streamlit server and produces
    # only "missing ScriptRunContext" warnings with no visible app. To
    # make a single click / a single `python app.py` invocation work
    # correctly everywhere -- including PyCharm -- this guard detects
    # that case and transparently re-launches the app under the real
    # Streamlit CLI in a subprocess.
    try:
        import streamlit as st
    except ImportError:
        print(
            "ERROR: the 'streamlit' package is not installed in this Python "
            "environment.\nInstall the dependencies first, e.g.:\n"
            "    pip install -r requirements.txt\n"
            "then run this app with:\n"
            "    streamlit run app.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        running_under_streamlit = st.runtime.exists()
    except Exception:
        # Older Streamlit versions do not expose st.runtime.exists(); assume
        # bare-mode and fall through to the self-relaunch path below, which
        # is safe even if this check is imperfect.
        running_under_streamlit = False

    if running_under_streamlit:
        _run_streamlit_app()
    else:
        print(
            "This is a Streamlit app -- launching it with 'streamlit run' "
            "automatically (equivalent to running:\n"
            f"    streamlit run \"{os.path.abspath(__file__)}\"\n"
            "in a terminal). In PyCharm, you can also create a Run/Debug "
            "configuration of type 'Python' that runs the module "
            "'streamlit' with parameters 'run app.py' to avoid this "
            "extra step next time.\n"
        )
        import subprocess

        cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))
