from dataclasses import dataclass
import numpy as np

CONSTELLATION = np.array([
    1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j
], dtype=np.complex128) / np.sqrt(2)

@dataclass(frozen=True)
class Domain:
    snr_db: float = 18.0
    phase_rad: float = 0.0
    gain: float = 1.0
    iq_imbalance: float = 0.0
    fading_jitter: float = 0.03

def features(z: np.ndarray) -> np.ndarray:
    """Real-valued features used by the small neural receiver."""
    return np.column_stack([z.real, z.imag, np.abs(z), np.angle(z)])

def generate_frame(rng: np.random.Generator, n_symbols: int, pilot_fraction: float,
                   domain: Domain) -> dict:
    labels = rng.integers(0, 4, size=n_symbols)
    n_pilots = max(4, int(n_symbols * pilot_fraction))
    pilot_mask = np.zeros(n_symbols, dtype=bool)
    pilot_mask[np.linspace(0, n_symbols - 1, n_pilots, dtype=int)] = True
    labels[pilot_mask] = np.arange(pilot_mask.sum()) % 4
    tx = CONSTELLATION[labels]
    fading = domain.gain * (1 + domain.fading_jitter * rng.normal(size=n_symbols))
    h = fading * np.exp(1j * domain.phase_rad)
    distorted = h * tx
    distorted = distorted.real * (1 + domain.iq_imbalance) + 1j * distorted.imag * (1 - domain.iq_imbalance)
    signal_power = np.mean(np.abs(distorted) ** 2)
    noise_power = signal_power / (10 ** (domain.snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.normal(size=n_symbols) + 1j * rng.normal(size=n_symbols))
    rx = distorted + noise
    return {"x": features(rx), "y": labels, "pilot_mask": pilot_mask, "rx": rx}

def domain_schedule(frame: int, total: int) -> tuple[str, Domain]:
    third = max(1, total // 3)
    if frame < third:
        return "source", Domain(snr_db=18, phase_rad=0.08, gain=1.0, iq_imbalance=0.01)
    if frame < 2 * third:
        return "phase_shift", Domain(snr_db=13, phase_rad=0.72, gain=0.82, iq_imbalance=0.08)
    return "deep_shift", Domain(snr_db=9, phase_rad=-0.58, gain=0.65, iq_imbalance=-0.12, fading_jitter=0.10)

