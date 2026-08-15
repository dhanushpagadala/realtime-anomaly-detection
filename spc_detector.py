"""
spc_detector.py
----------------
Statistical Process Control for streaming univariate metrics.

Two classic SPC techniques, updated online (O(1) per event, no batch needed):

  1. EWMA control chart  - exponentially weighted moving mean/variance with
                            a k-sigma control limit. Good at catching sudden
                            level shifts (e.g. a fraud spike).
  2. CUSUM                - cumulative sum of standardized deviations. Good
                            at catching small, sustained drifts (e.g. a
                            gradually worsening supply delay) that a single
                            point-in-time z-score would miss.

Both are maintained per metric-key (e.g. per region, per category) so a
spike in one segment doesn't get diluted/hidden by the global average.
"""

from collections import defaultdict


class EWMAMonitor:
    def __init__(self, lam: float = 0.2, k_sigma: float = 3.5, warmup: int = 30):
        """
        lam      : smoothing factor (higher = more reactive to recent data)
        k_sigma  : number of std-devs beyond which a point is flagged
        warmup   : number of points to observe before flagging begins
                   (avoids false positives while the baseline is still forming)
        """
        self.lam = lam
        self.k_sigma = k_sigma
        self.warmup = warmup
        self._mean = defaultdict(float)
        self._var = defaultdict(float)
        self._n = defaultdict(int)

    def update(self, key: str, value: float):
        n = self._n[key]
        if n == 0:
            self._mean[key] = value
            self._var[key] = 0.0
        else:
            prev_mean = self._mean[key]
            self._mean[key] = self.lam * value + (1 - self.lam) * prev_mean
            diff = value - prev_mean
            self._var[key] = (1 - self.lam) * (self._var[key] + self.lam * diff ** 2)
        self._n[key] += 1

        std = self._var[key] ** 0.5
        flagged = False
        score = 0.0
        if self._n[key] > self.warmup and std > 1e-9:
            score = abs(value - self._mean[key]) / std
            flagged = score > self.k_sigma
        return flagged, score


class CUSUMMonitor:
    def __init__(self, k: float = 0.5, h: float = 5.0, warmup: int = 30):
        """
        k : slack/reference value (in std-devs) — how much drift to tolerate
        h : decision threshold (in std-devs) — cumulative drift that triggers a flag
        """
        self.k = k
        self.h = h
        self.warmup = warmup
        self._mean = defaultdict(float)
        self._m2 = defaultdict(float)   # for streaming variance (Welford)
        self._n = defaultdict(int)
        self._sh = defaultdict(float)   # upper cumulative sum
        self._sl = defaultdict(float)   # lower cumulative sum

    def update(self, key: str, value: float):
        n = self._n[key] + 1
        self._n[key] = n
        delta = value - self._mean[key]
        self._mean[key] += delta / n
        self._m2[key] += delta * (value - self._mean[key])
        std = (self._m2[key] / n) ** 0.5 if n > 1 else 0.0

        flagged = False
        score = 0.0
        if n > self.warmup and std > 1e-9:
            z = (value - self._mean[key]) / std
            self._sh[key] = max(0.0, self._sh[key] + z - self.k)
            self._sl[key] = max(0.0, self._sl[key] - z - self.k)
            score = max(self._sh[key], self._sl[key])
            flagged = score > self.h
            if flagged:
                # reset after firing so it can re-arm for the next drift episode
                self._sh[key] = 0.0
                self._sl[key] = 0.0
        return flagged, score


class SPCDetector:
    """Combines EWMA (spike detection) + CUSUM (drift detection) per metric."""

    def __init__(self, ewma_kwargs: dict = None, cusum_kwargs: dict = None):
        self.ewma = EWMAMonitor(**(ewma_kwargs or {}))
        self.cusum = CUSUMMonitor(**(cusum_kwargs or {}))

    def score(self, key: str, value: float):
        ewma_flag, ewma_score = self.ewma.update(key, value)
        cusum_flag, cusum_score = self.cusum.update(key, value)
        flagged = ewma_flag or cusum_flag
        # normalize combined score to roughly comparable scale
        combined = max(ewma_score / 3.5, cusum_score / 5.0)
        return {
            "flagged": flagged,
            "score": round(combined, 3),
            "ewma_flag": ewma_flag,
            "ewma_score": round(ewma_score, 3),
            "cusum_flag": cusum_flag,
            "cusum_score": round(cusum_score, 3),
        }
