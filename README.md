# synthetic-infra-datasets

Synthetic data generator for data centre telemetry. Built this because getting real sensor data under NDA takes months, and I needed something to train on in the meantime.

The outputs are designed to mimic real sensor patterns, including realistic noise, seasonal variation, correlated failures, and the kinds of messy gaps you get from actual DCIM systems.

## Overview

Data centres have a lot of sensors but almost nobody publishes clean datasets. This tool generates synthetic time series data for:

- **Power**: per-rack PDU readings, UPS telemetry, utility feed data
- **Thermal**: CRAC/CRAH inlet/outlet temps, hot aisle containment, ambient
- **Cooling**: chiller load, cooling tower flow rates, delta-T across the loop
- **IT load**: server CPU/GPU utilisation proxies (correlates with power draw)
- **Supply chain events**: lead times, hardware arrivals, capacity planning signals

The goal is not perfect realism, it's useful realism. The patterns should be hard enough that naive models fail on them.

## Features

- Configurable fleet size (rack count, row layout, redundancy topology)
- Correlated sensor groups - if chiller 1 degrades, downstream temps drift realistically
- Synthetic fault injection: cooling failures, PDU trips, partial rack failures
- Export to Parquet, CSV, or InfluxDB line protocol
- Deterministic seed mode for reproducible training sets
- Holiday/weekend load profiles (most DCs have real weekly seasonality)

## Tech Stack

- Python 3.11
- NumPy / Pandas for the core generation
- `scipy.signal` for noise profiles
- `pyarrow` for Parquet output
- PyYAML for fleet config files

## How to Run

```bash
git clone https://github.com/codingg23/synthetic-infra-datasets
cd synthetic-infra-datasets
pip install -r requirements.txt

# generate a 90-day dataset for a 500-rack facility
python main.py --config examples/mid_tier_dc.yaml --days 90 --output ./data/

# inject a cooling fault at day 45
python main.py --config examples/mid_tier_dc.yaml --days 90 --fault cooling_partial --fault-day 45
```

Config files define the facility topology. See `examples/` for templates.

## Output Schema

```
timestamp, rack_id, row_id, pdu_kw, inlet_temp_c, outlet_temp_c,
cpu_util_pct, gpu_util_pct, ups_load_pct, crac_id, crac_load_pct
```

Plus a separate supply chain events CSV with lead time and order signals.

## Results / Learnings

Used this to pre-train the power forecasting model. Main lesson: the hardest thing to get right is the correlation structure between sensors, not the individual series. A rack with high GPU utilisation should cause thermal drift in adjacent racks 3-8 minutes later depending on airflow direction. Getting that lag right is what made the synthetic data actually useful for pretraining.

Supply chain generator is still rough. It generates Poisson-distributed order arrivals with realistic lead time distributions but the correlation between vendor delays is not modelled well yet.

## TODO

- [ ] Liquid cooling topology support (direct-to-chip, rear-door HX)
- [ ] More fault modes: intermittent sensor dropout, stuck at values
- [ ] Validation suite to compare statistical properties against real data
- [ ] Network/optical telemetry generator
