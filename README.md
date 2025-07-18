# synthetic-infra-datasets

Realistic synthetic data generator for data centre modelling. Built this because getting real telemetry data under NDA takes months, and I needed something to train on immediately.

The outputs are designed to mimic actual sensor patterns — including realistic noise, seasonal variation, correlated failures, and the kind of messy gaps you get from real DCIM systems.

## Overview

Data centres are full of sensors but almost no one publishes clean datasets. This tool generates synthetic time-series data for:

- **Power**: per-rack PDU readings, UPS telemetry, utility feed data
- **Thermal**: CRAC/CRAH inlet/outlet temps, hot aisle containment, ambient
- **Cooling**: chiller load, cooling tower flow rates, delta-T across the loop
- **IT load**: server CPU/GPU utilisation proxies (correlates with power draw)
- **Supply chain events**: lead times, hardware arrivals, capacity planning signals

The goal isn't perfect realism — it's *useful* realism. The patterns should be hard enough that naive models fail on them.

## Features

- Configurable fleet size (rack count, row layout, redundancy topology)
- Correlated sensor groups — if chiller 1 degrades, downstream temps drift realistically
- Inject synthetic faults: cooling failures, PDU trips, partial rack failures
- Export to Parquet, CSV, or InfluxDB line protocol
- Deterministic seed mode for reproducible training sets
- Holiday/weekend load profiles built in (most DCs have real weekly seasonality)

## Tech Stack

- Python 3.11
- NumPy / Pandas for the core generation
- `scipy.signal` for realistic noise profiles
- `pyarrow` for Parquet output
- PyYAML for fleet config files

## How to Run

```bash
git clone https://github.com/aryan-veltora/synthetic-infra-datasets
cd synthetic-infra-datasets
pip install -r requirements.txt

# generate a 90-day dataset for a 500-rack facility
python main.py --config examples/mid_tier_dc.yaml --days 90 --output ./data/
```

Config files let you define the facility topology. See `examples/` for a few templates.

```bash
# inject a cooling fault at day 45 and see what the thermal cascade looks like
python main.py --config examples/mid_tier_dc.yaml --days 90 --fault cooling_partial --fault-day 45
```

## Output Schema

Generated CSVs follow this rough schema:

```
timestamp, rack_id, row_id, pdu_kw, inlet_temp_c, outlet_temp_c,
cpu_util_pct, gpu_util_pct, ups_load_pct, crac_id, crac_load_pct
```

Plus a separate supply chain events CSV with lead time and order signals.

## Results / Learnings

Used this to pre-train early versions of the power forecasting model. Main lesson: the hardest thing to get right is the *correlation structure* between sensors, not the individual series. A rack with high GPU utilisation should cause thermal drift in adjacent racks 3–8 minutes later depending on airflow direction. Getting that right made the synthetic data actually useful for pretraining.

Still rough in places — the supply chain generator especially. That one is kind of a placeholder for now, just generates Poisson-distributed order arrivals with realistic lead time distributions.

## TODO

- [ ] Add liquid cooling topology support (direct-to-chip, rear-door HX)
- [ ] More fault modes: intermittent sensor dropout, stuck-at values
- [ ] Proper validation suite to compare statistical properties against real data
- [ ] Generator for network/optical telemetry (not just power/thermal)

## Contributing

Issues and PRs welcome. Especially interested in people who have worked with real DCIM systems and can tell me what the synthetic data gets wrong.
