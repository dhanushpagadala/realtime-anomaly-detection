"""
main.py
-------
End-to-end run of the pipeline:

  1. Stream simulated transactions through the HybridAnomalyDetector.
  2. Score against ground-truth anomaly labels to compute precision, recall,
     false-positive rate, and detection latency.
  3. Export the full annotated stream + summary metrics as JSON for the
     live dashboard, and print a benchmark report to stdout.

Usage:
    python3 main.py --n 5000 --anomaly-rate 0.04 --threshold 0.55
"""

import argparse
import json
import time
from pathlib import Path

from simulator import TransactionStreamSimulator
from hybrid_detector import HybridAnomalyDetector

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def run(n: int, anomaly_rate: float, threshold: float, seed: int = 7):
    sim = TransactionStreamSimulator(anomaly_rate=anomaly_rate, seed=seed)
    detector = HybridAnomalyDetector(alert_threshold=threshold)

    results = []
    t0 = time.perf_counter()
    for txn in sim.stream(n):
        r = detector.process(txn.to_dict())
        results.append(r)
    elapsed = time.perf_counter() - t0

    metrics = benchmark(results, elapsed)
    export(results, metrics, detector)
    print_report(metrics, n, elapsed)
    return results, metrics


def benchmark(results, elapsed):
    tp = fp = fn = tn = 0
    latencies_ms = []
    for r in results:
        truth = r["is_anomaly"]
        # "positive" = surfaced as an alert to a human (post alert-fatigue control),
        # since that's the outcome that actually matters operationally
        pred = r["alerted"]
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n = len(results)
    per_event_ms = (elapsed / n) * 1000 if n else 0.0

    # also report "raw detect" recall (before alert-fatigue suppression),
    # i.e. did the *math* catch it, independent of whether an alert fired
    raw_tp = sum(1 for r in results if r["is_anomaly"] and r["raw_flag"])
    raw_recall = raw_tp / (tp + fn + (raw_tp - tp)) if results else 0.0
    detected_but_suppressed = sum(1 for r in results if r["is_anomaly"] and r["suppressed"])

    by_type = {}
    for r in results:
        if r["is_anomaly"]:
            t = r["anomaly_type"]
            by_type.setdefault(t, {"total": 0, "caught": 0})
            by_type[t]["total"] += 1
            if r["alerted"]:
                by_type[t]["caught"] += 1

    return {
        "n_events": n,
        "n_true_anomalies": tp + fn,
        "n_alerts_fired": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "raw_detection_recall_pre_dedup": round((tp + detected_but_suppressed) / (tp + fn), 4) if (tp + fn) else 0.0,
        "detected_but_suppressed_by_fatigue_control": detected_but_suppressed,
        "avg_per_event_latency_ms": round(per_event_ms, 4),
        "total_runtime_sec": round(elapsed, 3),
        "by_anomaly_type": by_type,
    }


def export(results, metrics, detector):
    stream_path = OUT_DIR / "stream_export.json"
    metrics_path = OUT_DIR / "benchmark_metrics.json"

    # keep dashboard payload light: cap to last 1500 events for smooth animation
    payload = results[-1500:] if len(results) > 1500 else results
    with open(stream_path, "w") as f:
        json.dump(payload, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def print_report(metrics, n, elapsed):
    print("=" * 60)
    print("REAL-TIME ANOMALY DETECTION — BENCHMARK REPORT")
    print("=" * 60)
    print(f"Events processed:         {metrics['n_events']}")
    print(f"True anomalies (ground truth): {metrics['n_true_anomalies']}")
    print(f"Alerts fired (post-fatigue-control): {metrics['n_alerts_fired']}")
    print("-" * 60)
    print(f"Precision:                {metrics['precision']:.2%}")
    print(f"Recall:                   {metrics['recall']:.2%}")
    print(f"F1 score:                 {metrics['f1_score']:.3f}")
    print(f"False positive rate:      {metrics['false_positive_rate']:.2%}")
    print("-" * 60)
    print(f"Raw detector recall (pre alert-fatigue dedup): "
          f"{metrics['raw_detection_recall_pre_dedup']:.2%}")
    print(f"True anomalies caught but suppressed (cooldown): "
          f"{metrics['detected_but_suppressed_by_fatigue_control']}")
    print("-" * 60)
    print(f"Avg per-event latency:    {metrics['avg_per_event_latency_ms']:.4f} ms")
    print(f"Total runtime:            {metrics['total_runtime_sec']}s for {n} events")
    print("-" * 60)
    print("By anomaly type (caught / total):")
    for t, v in metrics["by_anomaly_type"].items():
        rate = v["caught"] / v["total"] if v["total"] else 0
        print(f"  {t:<16} {v['caught']:>4} / {v['total']:<4} ({rate:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--anomaly-rate", type=float, default=0.04)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run(args.n, args.anomaly_rate, args.threshold, args.seed)
