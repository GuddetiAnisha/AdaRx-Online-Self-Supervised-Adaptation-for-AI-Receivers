"""Simple constellation-centroid receiver for equalized 16-QAM payloads."""

from __future__ import annotations
import numpy as np

class CentroidReceiver:
    def __init__(self, n_classes=16):
        self.n_classes = int(n_classes)
        self.centroids = np.zeros(self.n_classes, dtype=np.complex128)
        self.counts = np.zeros(self.n_classes, dtype=float)
        self.anchor = self.centroids.copy()

    def fit(self, z, y):
        z = np.asarray(z, dtype=np.complex128)
        y = np.asarray(y, dtype=int)
        for c in range(self.n_classes):
            m = y == c
            if np.any(m):
                self.centroids[c] = z[m].mean()
                self.counts[c] = m.sum()
        if np.any(self.counts == 0):
            raise ValueError("source training data did not contain all 16 classes")
        self.anchor = self.centroids.copy()
        return self

    def distances(self, z):
        z = np.asarray(z, dtype=np.complex128)
        return np.abs(z[:, None] - self.centroids[None, :])

    def predict(self, z):
        return self.distances(z).argmin(axis=1)

    def confidence(self, z):
        d = self.distances(z)
        part = np.partition(d, 1, axis=1)
        d1, d2 = part[:, 0], part[:, 1]
        return np.clip(1.0 - d1 / (d2 + 1e-9), 0.0, 1.0)

    def adapt(self, z, confidence=0.45, rate=0.01, max_samples=256):
        z = np.asarray(z, dtype=np.complex128)
        pred = self.predict(z)
        conf = self.confidence(z)
        idx = np.flatnonzero(conf >= confidence)
        if len(idx) > max_samples:
            idx = idx[np.argsort(conf[idx])[-max_samples:]]
        for i in idx:
            c = int(pred[i])
            self.centroids[c] = (1-rate) * self.centroids[c] + rate * z[i]
        return int(len(idx)), float(np.mean(conf) if len(conf) else 0.0)

    def drift(self):
        return float(np.linalg.norm(self.centroids - self.anchor))
