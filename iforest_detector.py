"""
iforest_detector.py
--------------------
Isolation Forest adapted for a streaming context.

Isolation Forest is natively a batch algorithm (it needs a training set to
build its trees), so "streaming" support here means:

  1. Maintain a rolling window of the most recent N feature vectors.
  2. Periodically (every `retrain_every` events) refit the forest on that
     window. Because true anomalies are rare, the window is overwhelmingly
     "normal" traffic, so the forest's notion of normal adapts to concept
     drift (e.g. traffic mix shifting through the day) without manual
     re-tuning.
  3. Between refits, score new points against the *current* forest in O(log n)
     — this is what keeps per-event latency low enough to call "real-time".

Multivariate features (amount, quantity, unit_price, fulfillment latency)
let this catch anomalies that a univariate SPC monitor would miss — e.g. a
transaction where no single field is extreme, but the *combination* is
implausible (huge quantity of an expensive item shipped implausibly fast).
"""

from collections import deque
import numpy as np
from sklearn.ensemble import IsolationForest


FEATURE_NAMES = ["amount", "quantity", "unit_price", "fulfillment_latency_hrs"]


class StreamingIsolationForest:
    def __init__(self, window_size: int = 500, retrain_every: int = 50,
                 contamination: float = 0.05, min_train: int = 100, seed: int = 42):
        self.window_size = window_size
        self.retrain_every = retrain_every
        self.contamination = contamination
        self.min_train = min_train
        self.seed = seed

        self.buffer = deque(maxlen=window_size)
        self.model = None
        self._since_retrain = 0
        self._mean = None
        self._std = None

    def _vectorize(self, txn: dict):
        return np.array([txn[f] for f in FEATURE_NAMES], dtype=float)

    def _standardize(self, x: np.ndarray):
        if self._mean is None:
            return x
        std = np.where(self._std < 1e-9, 1.0, self._std)
        return (x - self._mean) / std

    def _retrain(self):
        data = np.array(self.buffer)
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)
        std = np.where(self._std < 1e-9, 1.0, self._std)
        norm = (data - self._mean) / std

        self.model = IsolationForest(
            n_estimators=60,
            max_samples=min(256, len(data)),
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=1,   # single small in-memory fit; process-pool overhead costs more than it saves
        )
        self.model.fit(norm)

    def update_and_score(self, txn: dict):
        x = self._vectorize(txn)
        self.buffer.append(x)
        self._since_retrain += 1

        # (re)train once we have enough history, then periodically after that
        if self.model is None and len(self.buffer) >= self.min_train:
            self._retrain()
            self._since_retrain = 0
        elif self.model is not None and self._since_retrain >= self.retrain_every:
            self._retrain()
            self._since_retrain = 0

        if self.model is None:
            return {"flagged": False, "score": 0.0, "ready": False}

        xn = self._standardize(x).reshape(1, -1)
        # decision_function: higher = more normal. Flip sign so higher = more anomalous.
        raw = -self.model.decision_function(xn)[0]
        pred = self.model.predict(xn)[0]  # -1 = anomaly, 1 = normal
        # squash raw isolation score (~[-0.5, 0.5]) into a friendlier 0-1ish range
        score = float(1 / (1 + np.exp(-8 * raw)))
        return {"flagged": pred == -1, "score": round(score, 3), "ready": True}
