"""
main.py  -  entry point for the synthetic data generator

Usage:
    python main.py --config examples/mid_tier_dc.yaml --days 90 --output ./data/
    python main.py --config examples/mid_tier_dc.yaml --days 90 --fault cooling_partial --fault-day 45
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import pandas as pd

from src.generators.power_generator import PowerGenerator, PowerGenConfig
from src.generators.thermal_generator import ThermalGenerator, ThermalConfig
from src.generators.supply_chain_generator import SupplyChainGenerator, SupplyChainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Synthetic DC data generator")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--output", default="./data/")
    parser.add_argument("--fault", choices=["cooling_partial", "pdu_trip", "none"], default="none")
    parser.add_argument("--fault-day", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = args.start
    end = pd.Timestamp(start) + pd.Timedelta(days=args.days)
    end_str = end.strftime("%Y-%m-%d")

    logger.info(f"Generating data from {start} to {end_str}")
    logger.info(f"Config: {args.config}, Fault: {args.fault}")

    # power
    power_config = PowerGenConfig(
        n_racks=cfg.get("n_racks", 100),
        rows=cfg.get("rows", 10),
        seed=args.seed,
    )
    power_gen = PowerGenerator(power_config)
    power_df = power_gen.generate(start=start, end=end_str)

    # thermal  -  uses power as input
    thermal_config = ThermalConfig(seed=args.seed + 1)
    thermal_gen = ThermalGenerator(thermal_config)
    thermal_df = thermal_gen.generate(power_df)

    # supply chain
    sc_config = SupplyChainConfig(seed=args.seed + 2)
    sc_gen = SupplyChainGenerator(sc_config)
    sc_df = sc_gen.generate(start=start, end=end_str)

    # save
    def save(df: pd.DataFrame, name: str):
        if args.format == "parquet":
            path = output_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        logger.info(f"Saved {name}: {len(df):,} rows → {path}")

    save(power_df, "power")
    save(thermal_df, "thermal")
    save(sc_df, "supply_chain")

    logger.info("Done.")


if __name__ == "__main__":
    main()
