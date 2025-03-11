"""
supply_chain_generator.py

Generates synthetic supply chain event data for data centre hardware.

This one is rougher than the thermal/power generators — supply chain
data is really hard to get and I haven't had as many real examples
to validate against. The main things I'm trying to capture:

  1. Lead times for servers/GPUs/networking are long and variable
     (especially during supply crunches — 2022-2023 was wild)
  2. There's a lumpy, project-driven demand pattern — capacity
     additions come in big chunks, not continuous trickle
  3. Procurement has a human in the loop, so you see things like
     orders that arrive faster than expected (expedited), orders
     that fall through, re-orders, etc.

For Veltora's use case: we want to predict when capacity will be
available, so procurement teams can plan infrastructure timelines
better. The forecasting problem is basically: given current open
orders and historical lead times, when will X units arrive?

Right now this just generates the historical events. The forecasting
piece is in the power-forecaster repo.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Hardware categories and rough base lead times (in days)
# These are ballpark figures based on what I've read/heard from DC operators
HARDWARE_LEAD_TIMES = {
    "server_standard": {"mean": 45, "std": 15},
    "server_gpu": {"mean": 120, "std": 45},      # GPUs are the killer
    "server_gpu_custom": {"mean": 180, "std": 60},
    "networking_switch_25g": {"mean": 30, "std": 10},
    "networking_switch_400g": {"mean": 90, "std": 30},
    "pdu_basic": {"mean": 20, "std": 7},
    "pdu_smart": {"mean": 45, "std": 15},
    "ups_module": {"mean": 60, "std": 20},
    "cooling_crac": {"mean": 75, "std": 25},
    "cooling_precision_ac": {"mean": 90, "std": 30},
}


@dataclass
class SupplyChainConfig:
    # how many procurement events to generate per year
    orders_per_year: int = 80
    # fraction that get expedited (vendor pushes harder, shorter lead time)
    expedite_rate: float = 0.12
    # fraction that slip beyond original estimate
    slip_rate: float = 0.25
    # fraction cancelled/re-ordered
    cancel_rate: float = 0.05
    # supply crunch period — lead times multiply during this window
    crunch_start: Optional[str] = "2024-06-01"
    crunch_end: Optional[str] = "2024-11-01"
    crunch_multiplier: float = 1.6
    seed: Optional[int] = None


class SupplyChainGenerator:

    def __init__(self, config: SupplyChainConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _in_crunch_period(self, order_date: pd.Timestamp) -> bool:
        if not self.config.crunch_start:
            return False
        crunch_start = pd.Timestamp(self.config.crunch_start)
        crunch_end = pd.Timestamp(self.config.crunch_end)
        return crunch_start <= order_date <= crunch_end

    def _sample_lead_time(self, hardware_type: str, order_date: pd.Timestamp) -> int:
        params = HARDWARE_LEAD_TIMES.get(hardware_type, {"mean": 60, "std": 20})
        lead_days = self.rng.normal(params["mean"], params["std"])
        lead_days = max(lead_days, 7)  # minimum 1 week regardless

        if self._in_crunch_period(order_date):
            lead_days *= self.config.crunch_multiplier

        return int(lead_days)

    def generate(self, start: str, end: str) -> pd.DataFrame:
        """
        Generate supply chain event log.

        Returns DataFrame with columns:
            order_id, order_date, hardware_type, quantity, vendor,
            quoted_lead_days, actual_lead_days, delivery_date,
            status, expedited, slipped, project_id
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        total_days = (end_ts - start_ts).days
        n_orders = int(self.config.orders_per_year * total_days / 365)

        logger.info(f"Generating {n_orders} supply chain events")

        hardware_types = list(HARDWARE_LEAD_TIMES.keys())
        vendors = ["Dell", "HPE", "Supermicro", "Arista", "Cisco", "Vertiv", "Schneider", "Lenovo"]

        # capacity expansion projects drive lumpy ordering
        # generate a few project milestones that trigger big orders
        n_projects = max(1, total_days // 120)
        project_dates = [
            start_ts + timedelta(days=int(self.rng.uniform(0, total_days)))
            for _ in range(n_projects)
        ]

        records = []
        for i in range(n_orders):
            # mostly random, but some orders cluster around project dates
            if self.rng.random() < 0.4 and project_dates:
                proj_date = project_dates[int(self.rng.integers(0, len(project_dates)))]
                jitter = timedelta(days=int(self.rng.normal(0, 14)))
                order_date = proj_date + jitter
                project_id = f"PROJ-{project_dates.index(proj_date):03d}"
            else:
                order_date = start_ts + timedelta(days=int(self.rng.uniform(0, total_days)))
                project_id = None

            order_date = max(start_ts, min(order_date, end_ts))
            hw_type = hardware_types[int(self.rng.integers(0, len(hardware_types)))]
            vendor = vendors[int(self.rng.integers(0, len(vendors)))]

            quoted_lead = self._sample_lead_time(hw_type, order_date)

            # apply modifiers
            status = "delivered"
            expedited = False
            slipped = False

            if self.rng.random() < self.config.cancel_rate:
                status = "cancelled"
                actual_lead = None
                delivery_date = None
            else:
                actual_lead = quoted_lead
                if self.rng.random() < self.config.expedite_rate:
                    expedited = True
                    actual_lead = int(actual_lead * self.rng.uniform(0.6, 0.85))
                elif self.rng.random() < self.config.slip_rate:
                    slipped = True
                    actual_lead = int(actual_lead * self.rng.uniform(1.1, 1.8))

                delivery_date = order_date + timedelta(days=actual_lead)
                if delivery_date > end_ts:
                    status = "in_transit"

            quantity = int(self.rng.choice([1, 2, 4, 8, 16, 32], p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05]))

            records.append({
                "order_id": f"PO-{i:05d}",
                "order_date": order_date.date(),
                "hardware_type": hw_type,
                "vendor": vendor,
                "quantity": quantity,
                "quoted_lead_days": quoted_lead,
                "actual_lead_days": actual_lead,
                "delivery_date": delivery_date.date() if delivery_date else None,
                "status": status,
                "expedited": expedited,
                "slipped": slipped,
                "project_id": project_id,
            })

        df = pd.DataFrame(records)
        df = df.sort_values("order_date").reset_index(drop=True)
        logger.info(f"Generated {len(df)} supply chain events ({df['status'].value_counts().to_dict()})")
        return df
