# Real-time anomaly detection for operational metrics

A streaming analytics pipeline that catches operational anomalies — fraud,
supply delays, and pricing errors — as transactions happen, instead of
waiting on a daily batch job.

## Why streaming instead of batch

A daily batch job means a pricing error sits live for up to 24 hours before
anyone notices, and a fraud burst has already completed before it's even
looked at. This pipeline scores every event as it arrives (~10ms/event) so
the detection lag drops from "next day" to "next few events."

## Architecture

```
simulator.py             synthetic transaction stream + injected ground-truth anomalies
spc_detector.py           EWMA + CUSUM statistical process control (univariate, per-segment)
iforest_detector.py       streaming isolation forest (multivariate, rolling retrain)
hybrid_detector.py        ensemble scoring + alert-fatigue control (cooldown/dedup)
main.py                   runs the stream end-to-end, benchmarks, exports dashboard data
dashboard/dashboard.html  live-replay dashboard (self-contained, no server needed)
```

**Detection is a hybrid of two complementary techniques:**

- **Statistical process control (EWMA + CUSUM)**, run per-segment (per region
  for amount/latency, per SKU for price against its catalog price). EWMA
  catches sudden spikes; CUSUM catches slow, sustained drift that a single
  z-score would miss. Segmenting the baseline matters — pooling all regions
  or all SKUs into one global average dilutes a real anomaly into noise.
- **A streaming isolation forest** over the full feature vector (amount,
  quantity, unit price, fulfillment latency), refit on a rolling window
  every 100 events. This catches anomalies where no single field is extreme
  but the *combination* is implausible — e.g. a large quantity of an
  expensive item shipped implausibly fast.

The two scores combine into a weighted ensemble score; alerts fire above a
tuned threshold, then pass through a per-user cooldown so a burst from one
source surfaces as one alert, not twenty (the alert-fatigue control).

## Results (8,000-event benchmark, 4% injected anomaly rate)

| Metric | Value |
|---|---|
| Precision | 93.6% |
| Recall | 75.8% |
| F1 | 0.838 |
| False positive rate | 0.22% |
| Avg per-event latency | 10.4 ms |

By anomaly type (caught / total): fraud 94/101 (93%), supply delay 101/108
(94%), pricing error 53/118 (45% — the hardest case, since a mispriced SKU
only stands out against its own catalog price history, not a category-wide
range).

Re-run `python3 src/main.py --n 8000` to reproduce; sweep `--threshold` to
see the precision/recall/FPR trade-off (0.5 → higher recall, more noise;
0.65 → very clean, fewer catches).

## Running it

```bash
pip install scikit-learn pandas numpy
cd src
python3 main.py --n 8000 --anomaly-rate 0.04 --threshold 0.6
```

This prints a benchmark report and writes `outputs/benchmark_metrics.json` +
`outputs/stream_export.json`. Open `dashboard/dashboard.html` directly in a
browser — no server needed — for a live-replay view of the stream with a
scrolling anomaly-score trace and an alert feed.

## What's simulated vs. real

The transaction stream, catalog, and ground-truth anomaly labels are
synthetic — built so precision/recall/FPR can be measured against a known
answer, which isn't possible on unlabeled real operational data. The
detection logic itself (EWMA, CUSUM, isolation forest, ensemble scoring,
cooldown-based alerting) is production-shaped: swap `simulator.py`'s
generator for a real Kafka/Kinesis consumer and the rest of the pipeline
runs unchanged.
