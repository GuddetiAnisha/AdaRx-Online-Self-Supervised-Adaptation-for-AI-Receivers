import unittest
import numpy as np
from adarx.equalization import estimate_channel, extract_payload_from_grid, PILOT_LOCAL_INDICES, PILOT_SYMBOL_INDEX, PILOT_SYMBOLS
from adarx.centroid import CentroidReceiver

class EqualizationTests(unittest.TestCase):
    def test_identity_channel(self):
        grid = np.zeros((14,128), dtype=np.complex128)
        for idx, s in zip(PILOT_LOCAL_INDICES, PILOT_SYMBOLS):
            grid[PILOT_SYMBOL_INDEX, 13+idx] = s
        h = estimate_channel(grid)
        self.assertTrue(np.allclose(h, 1.0+0j, atol=1e-6))

    def test_payload_count(self):
        active = np.ones((14,102), dtype=np.complex128)
        p = extract_payload_from_grid(active)
        self.assertEqual(p.shape, (1400,))

    def test_centroid_receiver(self):
        rng = np.random.default_rng(1)
        z, y = [], []
        for c in range(16):
            center = (c%4) + 1j*(c//4)
            pts = center + 0.01*(rng.normal(size=20) + 1j*rng.normal(size=20))
            z.extend(pts)
            y.extend([c]*20)
        m = CentroidReceiver().fit(np.asarray(z), np.asarray(y))
        pred = m.predict(np.asarray(z))
        self.assertGreater((pred == np.asarray(y)).mean(), 0.99)

if __name__ == "__main__":
    unittest.main()
