"""
thermal_generator.py

Generates correlated thermal sensor data from power draw time-series.

Physics-lite approach: treat each rack as a heat source, model airflow
propagation as a simple first-order lag from power to temperature.
Not CFD  -  just enough to make the correlations look real.

The key insight: temperature doesn't respond instantly to load changes.
There's a thermal mass in the aisle, rack, and servers. So a power spike
shows up in inlet/outlet temps ~3-10 minutes later depending on where
you're measuring. Getting that lag right was the main thing that made
the synthetic data useful.

TODO: hot aisle containment vs cold aisle containment have really
different profiles. Right now this is basically just open aisle.
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThermalConfig:
    # target inlet temp  -  ASHRAE A2 class is 10-35°C
    target_inlet_c: float = 21.0
    inlet_variation: float = 2.0      # ±°C variation around target
    # delta-T across rack at full load (outlet - inlet)
    max_delta_t: float = 15.0         # typical for high-density
    # thermal lag constant (in timesteps at 1-min freq)
    thermal_lag_steps: int = 5
    # CRAC/CRAH response lag
    cooling_response_steps: int = 12
    ambient_temp_c: float = 28.0      # outdoor ambient
    seed: Optional[int] = None


class ThermalGenerator:
    """
    Takes power draw data and generates correlated thermal readings.

    Inputs:
        power_df: output from PowerGenerator.generate()
        config: ThermalConfig

    Outputs:
        DataFrame with inlet_temp_c, outlet_temp_c, delta_t, crac_load_pct
    """

    def __init__(self, config: ThermalConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _apply_thermal_lag(self, signal: np.ndarray, lag_steps: int) -> np.ndarray:
        """
        First-order low-pass filter to simulate thermal mass.
        Rack temperature doesn't jump instantly when load changes.

        Using a simple IIR filter:
            y[t] = alpha * y[t-1] + (1-alpha) * x[t]
        where alpha controls the lag.
        """
        alpha = 1.0 - (1.0 / lag_steps)
        alpha = float(np.clip(alpha, 0.0, 0.99))
        b = [1 - alpha]
        a = [1.0, -alpha]
        return lfilter(b, a, signal)

    def _inlet_temp_profile(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Inlet temp has its own profile driven by CRAC setpoint and
        outdoor ambient. Simplified here  -  real cooling systems have
        complex control loops with deadbands, but this captures the
        general shape.
        """
        n = len(timestamps)
        hour = timestamps.hour.values

        # outdoor ambient drives a small diurnal variation in cooling efficiency
        # more pronounced in summer  -  hardcoding for now, should be configurable
        diurnal = 0.8 * np.sin(2 * np.pi * (hour - 14) / 24)  # peak at 2pm

        # slow drift in cooling setpoint  -  operators sometimes adjust this
        drift = 0.5 * np.sin(2 * np.pi * np.arange(n) / (n * 0.7))

        # sensor noise  -  inlet sensors are usually pretty good, low noise
        noise = self.rng.normal(0, 0.3, n)

        return self.config.target_inlet_c + diurnal + drift + noise

    def generate(self, power_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate thermal readings correlated with power_df.

        Expects power_df to have: timestamp, rack_id, row_id, pdu_kw
        Returns same granularity with thermal columns added.
        """
        result_parts = []

        racks = power_df["rack_id"].unique()
        timestamps = pd.DatetimeIndex(power_df["timestamp"].unique()).sort_values()

        inlet_base = self._inlet_temp_profile(timestamps)

        # build a dict for fast lookup
        power_pivot = power_df.pivot(index="timestamp", columns="rack_id", values="pdu_kw")
        max_kw_per_rack = power_df.groupby("rack_id")["pdu_kw"].max()

        logger.info(f"Generating thermal data for {len(racks)} racks")

        for rack_id in racks:
            if rack_id not in power_pivot.columns:
                logger.warning(f"Rack {rack_id} not found in power data, skipping")
                continue

            rack_power = power_pivot[rack_id].values
            rack_max = max_kw_per_rack.get(rack_id, 20.0)

            # normalised load fraction [0, 1]
            load_frac = rack_power / rack_max

            # thermal lag: outlet temp responds to load with a delay
            lagged_load = self._apply_thermal_lag(load_frac, self.config.thermal_lag_steps)

            # outlet = inlet + delta_T proportional to load
            outlet_delta = lagged_load * self.config.max_delta_t
            outlet_noise = self.rng.normal(0, 0.5, len(timestamps))

            inlet_temp = inlet_base + self.rng.normal(0, 0.2, len(timestamps))
            outlet_temp = inlet_temp + outlet_delta + outlet_noise

            # CRAC load  -  responds to aggregate row heat load with its own lag
            # oversimplified but directionally correct
            crac_load_raw = self._apply_thermal_lag(load_frac * 80 + 10, self.config.cooling_response_steps)
            crac_load = np.clip(crac_load_raw + self.rng.normal(0, 1.5, len(timestamps)), 5, 100)

            rack_thermal = pd.DataFrame({
                "timestamp": timestamps,
                "rack_id": rack_id,
                "inlet_temp_c": np.round(inlet_temp, 2),
                "outlet_temp_c": np.round(outlet_temp, 2),
                "delta_t_c": np.round(outlet_temp - inlet_temp, 2),
                "crac_load_pct": np.round(crac_load, 1),
            })
            result_parts.append(rack_thermal)

        df = pd.concat(result_parts, ignore_index=True)
        logger.info(f"Generated {len(df):,} thermal readings")
        return df
