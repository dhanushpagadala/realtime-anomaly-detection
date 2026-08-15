"""
hybrid_detector.py
-------------------
Wires the SPC monitors and the Streaming Isolation Forest together into a
single per-event anomaly score, then applies alerting logic that trades off
sensitivity against alert fatigue:

  - SPC runs per-segment on the metrics most tied to specific anomaly types
    (amount -> fraud/pricing, fulfillment latency -> supply delay).
  - Isolation Forest runs on the full multivariate feature vector to catch
    combinations SPC would miss.
  - A weighted ensemble score combines both; an event is flagged only if the
    combined score clears `alert_threshold`.
  - Alert fatigue control: once a (user_id, region) pair fires, it enters a
    cooldown window during which further flags are suppressed (logged as
    "detected but suppressed", not silently dropped) — this is what keeps
    the false-positive *alert* rate manageable even if the underlying
    detectors are noisy.
"""

from collections import defaultdict
from datetime import datetime

from spc_detector import SPCDetector
from iforest_detector import StreamingIsolationForest


class HybridAnomalyDetector:
    def __init__(self, alert_threshold: float = 0.55, spc_weight: float = 0.5,
                 iforest_weight: float = 0.5, cooldown_events: int = 25):
        self.spc_amount = SPCDetector()   # keyed by region -> catches fraud spikes
        self.spc_latency = SPCDetector()  # keyed by region -> catches supply delays
        # keyed by SKU -> many keys, each seen less often, so warm up fast;
        # legitimate per-SKU price variance is tiny (~3%), so even a short
        # baseline gives a reliable estimate and errors (6-20x or 0.01-0.15x) stand out clearly
        self.spc_price = SPCDetector(
            ewma_kwargs={"lam": 0.3, "k_sigma": 3.0, "warmup": 8},
            cusum_kwargs={"k": 0.5, "h": 4.0, "warmup": 8},
        )
        self.iforest = StreamingIsolationForest(window_size=400, retrain_every=100,
                                                  contamination=0.05, min_train=120)

        self.alert_threshold = alert_threshold
        self.spc_weight = spc_weight
        self.iforest_weight = iforest_weight

        self.cooldown_events = cooldown_events
        self._last_alert_event = defaultdict(lambda: -10_000)
        self._event_counter = 0

        self.alert_log = []          # alerts actually surfaced to the dashboard
        self.suppressed_log = []     # flags detected but held back (fatigue control)

    def process(self, txn: dict) -> dict:
        self._event_counter += 1

        amt_result = self.spc_amount.score(txn["region"], txn["amount"])
        lat_result = self.spc_latency.score(txn["region"], txn["fulfillment_latency_hrs"])
        price_result = self.spc_price.score(txn["sku"], txn["unit_price"])
        if_result = self.iforest.update_and_score(txn)

        spc_score = max(amt_result["score"], lat_result["score"], price_result["score"])
        combined = self.spc_weight * spc_score + self.iforest_weight * if_result["score"]
        combined = round(min(combined, 1.0), 3)

        raw_flag = combined >= self.alert_threshold

        # likely cause, for a human-readable alert reason
        reason = self._infer_reason(amt_result, lat_result, price_result, if_result)

        result = {
            **txn,
            "spc_amount_score": amt_result["score"],
            "spc_latency_score": lat_result["score"],
            "spc_price_score": price_result["score"],
            "iforest_score": if_result["score"],
            "combined_score": combined,
            "raw_flag": raw_flag,
            "alerted": False,
            "suppressed": False,
            "likely_cause": reason if raw_flag else None,
        }

        if raw_flag:
            key = (txn["user_id"], txn["region"])
            last = self._last_alert_event[key]
            if self._event_counter - last >= self.cooldown_events:
                self._last_alert_event[key] = self._event_counter
                result["alerted"] = True
                self.alert_log.append(result)
            else:
                result["suppressed"] = True
                self.suppressed_log.append(result)

        return result

    @staticmethod
    def _infer_reason(amt_result, lat_result, price_result, if_result):
        candidates = []
        if amt_result["flagged"]:
            candidates.append(("fraud_amount_spike", amt_result["score"]))
        if price_result["flagged"]:
            candidates.append(("pricing_error", price_result["score"]))
        if lat_result["flagged"]:
            candidates.append(("supply_delay", lat_result["score"]))
        if if_result["flagged"]:
            candidates.append(("multivariate_outlier", if_result["score"]))
        if not candidates:
            return "borderline_combined_score"
        return max(candidates, key=lambda c: c[1])[0]
