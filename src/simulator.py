"""
simulator.py
------------
Simulates a real-time stream of operational transactions (e.g. an e-commerce /
logistics platform) and injects three classes of anomalies so the detection
pipeline has ground truth to be benchmarked against:

  1. FRAUD           - abnormal transaction amount / velocity spikes
  2. SUPPLY_DELAY     - abnormal fulfillment/delivery latency
  3. PRICING_ERROR    - unit price far outside the expected band for a SKU

Each record is yielded one at a time (a generator), mimicking a message
arriving off a queue (Kafka/Kinesis/etc.) rather than being read from a
static batch file.
"""

import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

REGIONS = ["US-EAST", "US-WEST", "EU-CENTRAL", "APAC-SOUTH", "LATAM"]
CATEGORIES = ["electronics", "grocery", "apparel", "home_goods", "pharma"]

# "true" underlying price bands per category — used only to generate the catalog
PRICE_BANDS = {
    "electronics": (25, 900),
    "grocery": (2, 60),
    "apparel": (8, 200),
    "home_goods": (10, 350),
    "pharma": (5, 150),
}

SKUS_PER_CATEGORY = 25


def build_catalog(seed: int = 1):
    """
    A fixed catalog of SKUs, each with its own true list price. Pricing-error
    detection in the real world works by comparing the price on a transaction
    to the *catalog* price for that specific SKU — not a category-wide range
    (a $30 phone charger and a $30 candle are both "normal", just for
    different products). This is what makes per-SKU SPC on price meaningful.
    """
    rng = random.Random(seed)
    catalog = {}
    for category, (lo, hi) in PRICE_BANDS.items():
        for i in range(SKUS_PER_CATEGORY):
            sku = f"{category[:3].upper()}-{i:03d}"
            catalog[sku] = {
                "category": category,
                "true_price": round(rng.uniform(lo, hi), 2),
            }
    return catalog


CATALOG = build_catalog()

# expected fulfillment latency (hours) per region, used as ground truth for delays
LATENCY_MEAN = {
    "US-EAST": 18, "US-WEST": 20, "EU-CENTRAL": 24, "APAC-SOUTH": 30, "LATAM": 36,
}


class AnomalyType(str, Enum):
    NONE = "none"
    FRAUD = "fraud"
    SUPPLY_DELAY = "supply_delay"
    PRICING_ERROR = "pricing_error"


@dataclass
class Transaction:
    event_id: int
    timestamp: str
    user_id: str
    region: str
    category: str
    sku: str
    quantity: int
    unit_price: float
    amount: float
    fulfillment_latency_hrs: float
    is_anomaly: bool = False            # ground truth label
    anomaly_type: str = AnomalyType.NONE.value

    def to_dict(self):
        return asdict(self)


class TransactionStreamSimulator:
    """
    Generator-style simulator. Call `.stream(n)` to iterate n events, or
    `.stream()` for an unbounded generator (real streaming use case).

    anomaly_rate controls the *ground truth* injection rate, independent of
    whatever the detector ends up flagging — this is what benchmark.py scores
    precision/recall against.
    """

    def __init__(self, anomaly_rate: float = 0.04, seed: int = 42, start_time: datetime = None):
        self.anomaly_rate = anomaly_rate
        self.rng = random.Random(seed)
        self.event_id = 0
        self.clock = start_time or datetime.utcnow()
        self.user_pool = [f"user_{i:05d}" for i in range(2000)]

    def _base_transaction(self) -> Transaction:
        region = self.rng.choice(REGIONS)
        sku = self.rng.choice(list(CATALOG.keys()))
        category = CATALOG[sku]["category"]
        true_price = CATALOG[sku]["true_price"]
        # normal transactions vary slightly around the catalog price (promos, taxes, etc.)
        unit_price = round(true_price * self.rng.uniform(0.95, 1.05), 2)
        quantity = self.rng.randint(1, 6)
        amount = round(unit_price * quantity, 2)
        latency = max(1.0, self.rng.gauss(LATENCY_MEAN[region], LATENCY_MEAN[region] * 0.15))

        self.event_id += 1
        self.clock += timedelta(seconds=self.rng.uniform(0.2, 2.5))

        return Transaction(
            event_id=self.event_id,
            timestamp=self.clock.isoformat(),
            user_id=self.rng.choice(self.user_pool),
            region=region,
            category=category,
            sku=sku,
            quantity=quantity,
            unit_price=unit_price,
            amount=amount,
            fulfillment_latency_hrs=round(latency, 2),
        )

    def _inject_anomaly(self, txn: Transaction) -> Transaction:
        kind = self.rng.choice(list(AnomalyType)[1:])  # exclude NONE

        if kind == AnomalyType.FRAUD:
            # sudden high-value / high-quantity spike, atypical for a single user
            txn.quantity = self.rng.randint(20, 80)
            txn.unit_price = round(txn.unit_price * self.rng.uniform(3, 8), 2)
            txn.amount = round(txn.unit_price * txn.quantity, 2)

        elif kind == AnomalyType.SUPPLY_DELAY:
            base = LATENCY_MEAN[txn.region]
            txn.fulfillment_latency_hrs = round(base * self.rng.uniform(4, 9), 2)

        elif kind == AnomalyType.PRICING_ERROR:
            true_price = CATALOG[txn.sku]["true_price"]
            # decimal/entry error relative to THIS SKU's catalog price:
            # e.g. $499 listed as $4.99 (decimal slip) or $4990 (extra zero)
            if self.rng.random() < 0.5:
                txn.unit_price = round(true_price * self.rng.uniform(0.01, 0.15), 2)
            else:
                txn.unit_price = round(true_price * self.rng.uniform(6, 20), 2)
            txn.amount = round(txn.unit_price * txn.quantity, 2)

        txn.is_anomaly = True
        txn.anomaly_type = kind.value
        return txn

    def next_event(self) -> Transaction:
        txn = self._base_transaction()
        if self.rng.random() < self.anomaly_rate:
            txn = self._inject_anomaly(txn)
        return txn

    def stream(self, n: int = None, realtime: bool = False, delay: float = 0.05):
        """
        Yields Transaction objects. If realtime=True, sleeps `delay` seconds
        between events to actually simulate wall-clock streaming (useful for
        a live demo); otherwise yields as fast as possible (useful for
        benchmarking against thousands of events).
        """
        count = 0
        while n is None or count < n:
            yield self.next_event()
            count += 1
            if realtime:
                time.sleep(delay)


if __name__ == "__main__":
    sim = TransactionStreamSimulator(anomaly_rate=0.05, seed=1)
    for txn in sim.stream(5):
        print(txn.to_dict())
