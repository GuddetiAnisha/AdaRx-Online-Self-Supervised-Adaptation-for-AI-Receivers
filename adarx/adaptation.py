import time
import numpy as np

def adapt(receiver, frame, method, lr=0.015, confidence=0.92):
    """Original synthetic-data adaptation routine."""
    if method == "static":
        return 0.0, 0.0, 0
    x, y, pilots = frame["x"], frame["y"], frame["pilot_mask"]
    train_x, train_y = x[pilots], y[pilots]
    weights = np.ones(len(train_y))
    anchor = 0.002
    if method == "hybrid":
        _, p = receiver.forward(x[~pilots])
        conf = p.max(1)
        pseudo = p.argmax(1)
        keep = conf >= confidence
        if keep.any():
            train_x = np.vstack([train_x, x[~pilots][keep]])
            train_y = np.concatenate([train_y, pseudo[keep]])
            weights = np.concatenate([weights, 0.35 * conf[keep]])
        anchor = 0.006
    start = time.perf_counter()
    loss = receiver.update(train_x, train_y, lr=lr, steps=2, anchor_strength=anchor, weights=weights)
    return loss, (time.perf_counter() - start) * 1000, len(train_y)


def pseudo_label_adapt(receiver, x, confidence=0.90, lr=0.005,
                       anchor_strength=0.01, max_samples=256):
    """Self-supervised update using only high-confidence model predictions.

    Ground-truth labels are not used for adaptation. This is the real-SDR
    streaming baseline because the public dataset's pilot symbols are QPSK-like
    reference tones while the payload task is 16-QAM classification.
    """
    start = time.perf_counter()
    _, p = receiver.forward(x)
    conf = p.max(1)
    pseudo = p.argmax(1)
    keep_idx = np.flatnonzero(conf >= confidence)
    if len(keep_idx) == 0:
        return 0.0, (time.perf_counter() - start) * 1000, 0
    if len(keep_idx) > max_samples:
        keep_idx = keep_idx[np.argsort(conf[keep_idx])[-max_samples:]]
    weights = conf[keep_idx]
    loss = receiver.update(
        x[keep_idx], pseudo[keep_idx], lr=lr, steps=1,
        anchor_strength=anchor_strength, weights=weights,
    )
    return loss, (time.perf_counter() - start) * 1000, int(len(keep_idx))
