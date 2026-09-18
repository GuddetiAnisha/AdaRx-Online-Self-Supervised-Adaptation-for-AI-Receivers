import tempfile
import unittest
import numpy as np
from adarx.channel import Domain, generate_frame
from adarx.model import NeuralReceiver
from adarx.adaptation import adapt
from adarx.data import load_field_csv

class AdaRxTests(unittest.TestCase):
    def test_frame_shapes(self):
        f=generate_frame(np.random.default_rng(1),128,.1,Domain())
        self.assertEqual(f["x"].shape,(128,4)); self.assertGreater(f["pilot_mask"].sum(),0)
    def test_training_reduces_loss(self):
        rng=np.random.default_rng(2); f=generate_frame(rng,512,.1,Domain(snr_db=30))
        m=NeuralReceiver(2); before=m.update(f["x"],f["y"],lr=.02,steps=1)
        after=m.update(f["x"],f["y"],lr=.02,steps=80)
        self.assertLess(after,before)
    def test_adaptation_updates(self):
        f=generate_frame(np.random.default_rng(3),128,.2,Domain())
        m=NeuralReceiver(3); _,_,n=adapt(m,f,"pilot"); self.assertEqual(n,int(f["pilot_mask"].sum()))
    def test_csv_adapter(self):
        with tempfile.NamedTemporaryFile("w",suffix=".csv",delete=False) as f:
            f.write("frame_id,rx_i,rx_q,tx_class,is_pilot\n0,1,1,0,1\n0,-1,1,1,0\n"); name=f.name
        frames=load_field_csv(name); self.assertEqual(frames[0]["x"].shape,(2,4))

if __name__=="__main__": unittest.main()

