# Bengaluru EV-RF Network Optimiser

Optimal siting and capacity planning of Electric Vehicle Recharging Facilities
(EV-RF) for Bengaluru, Karnataka, using an exact Mixed-Integer Linear Program
(MILP) solved via Branch-and-Bound (COIN-OR CBC), benchmarked against two
heuristic baselines, with a GeoJSON-driven Streamlit dashboard.
App URL: https://ev-rf-network-optimization.streamlit.app/
## Contents

- `app.py` — single-file Python application: data generation, demand
  forecasting, the MILP optimisation model, comparative baselines, GeoJSON
  export, and the full Streamlit UI (run with `streamlit run app.py`).
- `requirements.txt` — Python dependencies.
- `EV_RF_Bengaluru_Optimization_Paper.docx` — the accompanying research
  paper (Emerald-style structured abstract, literature review, full
  mathematical formulation, results, comparative analysis, spatial maps).
- `data/` — downloadable datasets and GeoJSON layers used to produce the
  paper's results (full 64x64 / 100-supply-point scale):
  - `Demand_Grid.csv`, `Demand_History.csv` (2010-2018), `Demand_Forecast.csv`
  - `Existing_EV_infrastructure_2018.csv`, `Optimised_EV_Infrastructure.csv`
  - `Bengaluru_Boundary.geojson`, `Demand_Points.geojson`,
    `Supply_Points_2018.geojson`, `Supply_Points_Optimised.geojson`
- `figures/` — the static map/chart figures embedded in the paper.
- `tables/` — the comparative-analysis and forecast-model-comparison tables
  (CSV) underlying the paper's Table 1 and Table 2.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

**PyCharm users:** you can also just click the "Run" button on `app.py`
directly (or run `python app.py` from a terminal). The script detects that
it was not launched via `streamlit run` and automatically re-launches
itself under the Streamlit server, so either route works with no extra
setup. To avoid the one-time re-launch message, you can instead create a
PyCharm Run/Debug configuration of type "Python" that runs the module
`streamlit` with parameters `run app.py`.

Then, in the sidebar: (1) Generate/regenerate data, (2) Run demand
forecasting, (3) Run optimisation (MILP + baselines). Use the tabs to explore
the model documentation, data, forecasts, KPIs, interactive GeoJSON maps, and
comparative analysis, and download any data product from the app.

The default grid resolution (24x24) is tuned for fast interactive use; the
paper's reported results use the full 64x64 / 100-supply-point configuration
with a longer MILP time budget (see the paper's Methodology section).

## Dataset and forecast horizon

Demand history now spans 2010 through the partial year January-June 2026
(sixteen full calendar years plus one partial-year observation), extended
from the original 2010-2018 window so that operational forecasts are
produced for full-year 2027 and 2028 instead of 2019/2020. The 2018
baseline infrastructure inventory (existing SCS/FCS counts and parking-slot
capacity at each of the 100 candidate sites) is deliberately held fixed
throughout, so the "incremental expansion only" constraint has a single,
unambiguous starting point -- see the paper's Methodology (Section 3.2) and
Results (Section 4.2) for what this reveals once forecast demand grows well
beyond the network's fixed physical capacity.

## Spatial maps: real place names and basemap choice

The Spatial Maps tab lets you choose a base map style (OpenStreetMap and
Esri World Street Map are free, keyless layers whose tiles already carry
real street and place-name cartography; two Google layers are also offered
using Google's raw, unauthenticated tile endpoint for quick evaluation --
see the in-app caption for why a proper Google Maps Platform API key and
the official Maps JavaScript/Static API are what a production deployment
should use instead). A "Show place-name labels" toggle overlays each EV-RF
site's real locality name permanently on the map itself (not just on
hover), so the map reads like a real, labelled map rather than an abstract
grid of coordinates.
