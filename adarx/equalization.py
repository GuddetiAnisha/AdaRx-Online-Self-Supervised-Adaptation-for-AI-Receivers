"""Pilot-aided OFDM channel estimation and equalization for the public SDR dataset."""

from __future__ import annotations
import numpy as np

FFT_SIZE = 128
N_SYMBOLS = 14
OFFSET = 13
ACTIVE_WIDTH = 102
DC_LOCAL_INDEX = 51
PILOT_SYMBOL_INDEX = 2
PILOT_LOCAL_INDICES = np.array(list(range(0, 97, 8)) + [101], dtype=int)

PILOT_SYMBOLS = np.array([
    -0.7-0.7j, -0.7+0.7j,  0.7-0.7j,  0.7+0.7j,
    -0.7-0.7j, -0.7+0.7j,  0.7-0.7j,  0.7+0.7j,
    -0.7-0.7j, -0.7+0.7j,  0.7-0.7j,  0.7+0.7j,
    -0.7-0.7j, -0.7+0.7j,
], dtype=np.complex128)

def active_grid(pdsch_iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(pdsch_iq)
    if iq.shape != (N_SYMBOLS, FFT_SIZE):
        raise ValueError(f"expected IQ shape {(N_SYMBOLS, FFT_SIZE)}, got {iq.shape}")
    return iq[:, OFFSET:OFFSET + ACTIVE_WIDTH].astype(np.complex128)

def estimate_channel(pdsch_iq: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    grid = active_grid(pdsch_iq)
    rx_pilots = grid[PILOT_SYMBOL_INDEX, PILOT_LOCAL_INDICES]
    h_pilot = rx_pilots / np.where(np.abs(PILOT_SYMBOLS) > eps, PILOT_SYMBOLS, 1.0)
    x = PILOT_LOCAL_INDICES.astype(float)
    target = np.arange(ACTIVE_WIDTH, dtype=float)
    h_real = np.interp(target, x, h_pilot.real)
    h_imag = np.interp(target, x, h_pilot.imag)
    h = h_real + 1j * h_imag
    small = np.abs(h) < 1e-3
    if np.any(small):
        phase = np.exp(1j * np.angle(h[small] + 1e-12))
        h[small] = 1e-3 * phase
    return h

def equalize_grid(pdsch_iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = active_grid(pdsch_iq)
    h = estimate_channel(pdsch_iq)
    return grid / h[None, :], h

def extract_payload_from_grid(grid: np.ndarray) -> np.ndarray:
    values = []
    pilots = set(int(i) for i in PILOT_LOCAL_INDICES)
    for sym in range(N_SYMBOLS):
        for sc in range(ACTIVE_WIDTH):
            if sc == DC_LOCAL_INDEX:
                continue
            if sym == PILOT_SYMBOL_INDEX and sc in pilots:
                continue
            values.append(grid[sym, sc])
    out = np.asarray(values, dtype=np.complex128)
    if len(out) != 1400:
        raise RuntimeError(f"payload mask produced {len(out)} symbols, expected 1400")
    return out

def equalized_payload(pdsch_iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eq, h = equalize_grid(pdsch_iq)
    return extract_payload_from_grid(eq), h

def channel_quality_features(h: np.ndarray) -> dict[str, float]:
    mag = np.abs(h)
    phase = np.unwrap(np.angle(h))
    return {
        "channel_mag_mean": float(np.mean(mag)),
        "channel_mag_std": float(np.std(mag)),
        "channel_phase_std": float(np.std(phase)),
        "channel_min_mag": float(np.min(mag)),
    }
