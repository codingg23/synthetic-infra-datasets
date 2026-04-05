"""
power_generator.py

Generates synthetic power draw time-series for data centre racks and PDUs.

The main challenge here is making the load look realistic  -  real DC power
has strong daily/weekly seasonality, correlated spikes across racks in the
same cluster, and occasional weird artefacts from UPS switchovers.

Started simple with a sin wave + noise, but that looked obviously fake.
The current approach layers in multiple components:
  1. Base load (facility always has a floor)
  2. Business-hours multiplier (weekdays > weekends, 9am-6pm > overnight)
  3. Batch job spikes (Poisson-distributed, cluster-correlated)
  4. Per-rack noise (AR(1) process so it's not just iid)
  5. Optional UPS transfer events
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RackConfig:
    rack_id: str
    row_id: str
    max_kw: float = 20.0          # typical high-density rack
    base_load_fraction: float = 0.35  # idle load as fraction of max
    cluster_id: Optional[str] = None  # racks in same cluster get correlated spikes


@dataclass
class PowerGenConfig:
    n_racks: int = 100
    rows: int = 10
    freq: str = "1min"            # sample frequency
    noise_ar_coef: float = 0.7    # autocorrelation in per-rack noise
    spike_rate: float = 0.02      # average batch job spikes per rack per hour
    spike_magnitude: float = 0.4  # fraction of headroom used during spike
    seed: Optional[int] = None


class PowerGenerator:
    """
    Generates per-rack power draw time-series.

    Usage:
        config = PowerGenConfig(n_racks=50, seed=42)
        gen = PowerGenerator(config)
        df = gen.generate(start="2024-01-01", end="2024-04-01")
    """

    def __init__(self, config: PowerGenConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._rack_configs = self._init_racks()

    def _init_racks(self) -> list[RackConfig]:
        racks = []
        rows_per_col = self.config.n_racks // self.config.rows

        for row_idx in range(self.config.rows):
            for rack_idx in range(rows_per_col):
                rack_id = f"R{row_idx:02d}-{rack_idx:02d}"
                row_id = f"ROW-{row_idx:02d}"
                # assign cluster  -  every 4 racks share a top-of-rack switch
                cluster_id = f"cluster_{(row_idx * rows_per_col + rack_idx) // 4}"

                # vary max_kw a bit  -  not all racks are identical
                max_kw = self.rng.normal(20.0, 3.0)
                max_kw = float(np.clip(max_kw, 8.0, 40.0))

                racks.append(RackConfig(
                    rack_id=rack_id,
                    row_id=row_id,
                    max_kw=max_kw,
                    base_load_fraction=self.rng.uniform(0.25, 0.50),
                    cluster_id=cluster_id,
                ))

        return racks

    def _business_hours_multiplier(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Business hours load profile.

        Real data centres don't have a hard 9-5 shape, but there IS a meaningful
        weekly pattern  -  batch jobs tend to run overnight and on weekends,
        interactive workloads peak during business hours.

        This is a rough approximation. The actual shape varies a lot by customer mix.
        """
        hour = timestamps.hour.values
        dow = timestamps.dayofweek.values  # 0=Monday

        # base daily profile  -  gentle curve, not a sharp step
        # peaks around 2pm, drops overnight
        daily = 0.7 + 0.3 * np.sin(np.pi * (hour - 6) / 16) ** 2
        daily = np.where((hour < 6) | (hour > 22), 0.75, daily)  # overnight floor

        # weekend dip
        is_weekend = (dow >= 5).astype(float)
        weekend_factor = 1.0 - 0.15 * is_weekend

        return daily * weekend_factor

    def _generate_ar1_noise(self, n_steps: int, coef: float, scale: float = 0.05) -> np.ndarray:
        """
        AR(1) noise so consecutive readings are correlated.
        Makes the series look like a real sensor (not iid).
        """
        noise = np.zeros(n_steps)
        eps = self.rng.normal(0, scale, n_steps)
        noise[0] = eps[0]
        for t in range(1, n_steps):
            noise[t] = coef * noise[t - 1] + eps[t]
        return noise

    def _generate_cluster_spikes(
        self,
        timestamps: pd.DatetimeIndex,
        cluster_ids: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Batch job spikes that are correlated within a cluster.
        When a job lands on a cluster, all racks in that cluster see
        a load jump  -  not perfectly synchronised, but close.
        """
        n_steps = len(timestamps)
        unique_clusters = list(set(cluster_ids))
        spike_series = {cid: np.zeros(n_steps) for cid in unique_clusters}

        dt_hours = (timestamps[1] - timestamps[0]).total_seconds() / 3600
        spike_prob = self.config.spike_rate * dt_hours  # per-step probability

        for cid in unique_clusters:
            # generate spike events for this cluster
            spike_mask = self.rng.random(n_steps) < spike_prob
            spike_indices = np.where(spike_mask)[0]

            for idx in spike_indices:
                # spike decays exponentially  -  batch jobs ramp up then finish
                duration = int(self.rng.exponential(30))  # ~30 steps average
                end_idx = min(idx + duration, n_steps)
                magnitude = self.config.spike_magnitude * self.rng.uniform(0.5, 1.0)

                # ramp up / plateau / ramp down shape
                spike_len = end_idx - idx
                if spike_len > 3:
                    ramp_up = np.linspace(0, magnitude, spike_len // 3)
                    plateau = np.full(spike_len - 2 * (spike_len // 3), magnitude)
                    ramp_down = np.linspace(magnitude, 0, spike_len // 3)
                    spike_shape = np.concatenate([ramp_up, plateau, ramp_down])
                else:
                    spike_shape = np.full(spike_len, magnitude)

                spike_series[cid][idx:end_idx] += spike_shape[:end_idx - idx]

        return spike_series

    def generate(self, start: str, end: str) -> pd.DataFrame:
        """
        Generate power draw time-series for all racks.

        Returns a tidy DataFrame with columns:
            timestamp, rack_id, row_id, pdu_kw, ups_load_pct
        """
        timestamps = pd.date_range(start=start, end=end, freq=self.config.freq)
        n_steps = len(timestamps)
        logger.info(f"Generating power data for {len(self._rack_configs)} racks, {n_steps} steps")

        biz_mult = self._business_hours_multiplier(timestamps)
        cluster_ids = [r.cluster_id for r in self._rack_configs]
        cluster_spikes = self._generate_cluster_spikes(timestamps, cluster_ids)

        records = []
        for rack in self._rack_configs:
            base = rack.base_load_fraction * rack.max_kw
            headroom = rack.max_kw - base

            # layer the components
            load = base + headroom * biz_mult  # business hours shape
            load += headroom * cluster_spikes[rack.cluster_id]  # batch spikes

            # per-rack AR(1) noise
            noise = self._generate_ar1_noise(n_steps, self.config.noise_ar_coef)
            load += headroom * 0.1 * noise

            # clip to physical limits (can't draw more than max, floor at ~10%)
            load = np.clip(load, rack.max_kw * 0.1, rack.max_kw)

            # UPS load is just a slight overhead above IT load
            # 95% efficiency assumption  -  rough but fine for now
            ups_load_pct = (load / rack.max_kw) * (1 / 0.95) * 100

            rack_df = pd.DataFrame({
                "timestamp": timestamps,
                "rack_id": rack.rack_id,
                "row_id": rack.row_id,
                "pdu_kw": np.round(load, 3),
                "ups_load_pct": np.round(np.clip(ups_load_pct, 0, 100), 1),
            })
            records.append(rack_df)

        df = pd.concat(records, ignore_index=True)
        logger.info(f"Generated {len(df):,} power readings")
        return df
