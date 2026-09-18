import csv
import numpy as np
from .channel import features

def load_field_csv(path):
    """Load a simple field-data exchange format without assuming Ericsson schemas."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"frame_id", "rx_i", "rx_q"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain {sorted(required)}")
    output = []
    for fid in sorted({r["frame_id"] for r in rows}, key=lambda x: int(x)):
        group = [r for r in rows if r["frame_id"] == fid]
        rx = np.array([float(r["rx_i"]) + 1j*float(r["rx_q"]) for r in group])
        y = np.array([int(r.get("tx_class", -1)) for r in group])
        pilots = np.array([r.get("is_pilot", "0").lower() in {"1","true","yes"} for r in group])
        output.append({"frame_id": int(fid), "x": features(rx), "rx": rx, "y": y, "pilot_mask": pilots})
    return output

