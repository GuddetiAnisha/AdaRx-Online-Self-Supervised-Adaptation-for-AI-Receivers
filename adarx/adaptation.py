import time
import numpy as np

def adapt(receiver, frame, method, lr=0.015, confidence=0.92):
    """Return update loss, milliseconds, and number of adaptation samples."""
    if method == "static": return 0.0, 0.0, 0
    x, y, pilots = frame["x"], frame["y"], frame["pilot_mask"]
    train_x, train_y = x[pilots], y[pilots]
    weights = np.ones(len(train_y))
    anchor = 0.002
    if method == "hybrid":
        _, p = receiver.forward(x[~pilots])
        conf = p.max(1); pseudo = p.argmax(1); keep = conf >= confidence
        if keep.any():
            train_x = np.vstack([train_x, x[~pilots][keep]])
            train_y = np.concatenate([train_y, pseudo[keep]])
            weights = np.concatenate([weights, 0.35 * conf[keep]])
        anchor = 0.006
    start = time.perf_counter()
    loss = receiver.update(train_x, train_y, lr=lr, steps=2, anchor_strength=anchor, weights=weights)
    return loss, (time.perf_counter() - start) * 1000, len(train_y)

