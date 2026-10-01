"""Adapter for rikluost/sdr_ofdm_dataset_v1 measured OFDM dataset."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from .channel import features
from .equalization import equalized_payload, channel_quality_features

FFT_SIZE = 128
N_SYMBOLS = 14
OFFSET = 13
ACTIVE_WIDTH = 102
DC_LOCAL_INDEX = 51
PILOT_SYMBOL_INDEX = 2
PILOT_LOCAL_INDICES = tuple(list(range(0, 97, 8)) + [101])

def bits_to_class(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 2 or bits.shape[1] != 4:
        raise ValueError("expected labels with shape [N,4]")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bit labels must contain only 0/1")
    return bits @ np.array([8,4,2,1], dtype=int)

def payload_iq_from_tti(pdsch_iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(pdsch_iq)
    if iq.shape != (N_SYMBOLS, FFT_SIZE):
        raise ValueError(f"expected IQ shape {(N_SYMBOLS, FFT_SIZE)}, got {iq.shape}")
    active = iq[:, OFFSET:OFFSET + ACTIVE_WIDTH]
    values = []
    pilots = set(PILOT_LOCAL_INDICES)
    for sym in range(N_SYMBOLS):
        for sc in range(ACTIVE_WIDTH):
            if sc == DC_LOCAL_INDEX:
                continue
            if sym == PILOT_SYMBOL_INDEX and sc in pilots:
                continue
            values.append(active[sym, sc])
    out = np.asarray(values, dtype=np.complex128)
    if len(out) != 1400:
        raise RuntimeError(f"payload mask produced {len(out)} symbols, expected 1400")
    return out

def sample_to_frame(pdsch_iq, labels, sinr, frame_id=0):
    raw_rx = payload_iq_from_tti(np.asarray(pdsch_iq))
    eq_rx, h = equalized_payload(np.asarray(pdsch_iq))
    bit_labels = np.asarray(labels)
    if bit_labels.shape != (1400, 4):
        raise ValueError(f"expected label shape (1400,4), got {bit_labels.shape}")
    y = bits_to_class(bit_labels)
    sinr_value = float(np.asarray(sinr).reshape(-1)[0])
    raw_x = features(raw_rx)
    eq_x = features(eq_rx)
    return {
        "frame_id": int(frame_id),
        "raw_rx": raw_rx,
        "eq_rx": eq_rx,
        "raw_x": raw_x,
        "eq_x": eq_x,
        "x": eq_x,
        "rx": eq_rx,
        "y": y,
        "sinr_db": sinr_value,
        **channel_quality_features(h),
    }

def load_sdr_pth(path: str | Path, max_ttis: int | None = None):
    import torch
    dataset = torch.load(str(path), map_location="cpu", weights_only=False)
    n = len(dataset) if max_ttis is None else min(len(dataset), int(max_ttis))
    return [sample_to_frame(*dataset[i], frame_id=i) for i in range(n)]
