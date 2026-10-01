import unittest
import numpy as np
from adarx.sdr_dataset import bits_to_class, payload_iq_from_tti, sample_to_frame
from adarx.model import NeuralReceiver
from adarx.metrics import real_sdr_metrics


class SDRDatasetTests(unittest.TestCase):
    def test_bits_to_class(self):
        bits=np.array([[0,0,0,0],[0,0,0,1],[1,1,1,1]])
        self.assertEqual(bits_to_class(bits).tolist(), [0,1,15])

    def test_payload_mask_count(self):
        iq=np.arange(14*128).reshape(14,128).astype(np.complex128)
        payload=payload_iq_from_tti(iq)
        self.assertEqual(payload.shape, (1400,))

    def test_sample_frame_shape(self):
        iq=np.ones((14,128),dtype=np.complex64)
        labels=np.zeros((1400,4),dtype=np.float32)
        f=sample_to_frame(iq,labels,np.array([12.0]),3)
        self.assertEqual(f["x"].shape,(1400,4))
        self.assertEqual(f["y"].shape,(1400,))
        self.assertEqual(f["frame_id"],3)

    def test_16_class_receiver(self):
        m=NeuralReceiver(seed=1,n_classes=16)
        x=np.random.default_rng(1).normal(size=(32,4))
        p=m.predict(x)
        self.assertTrue(np.all((p>=0)&(p<16)))

    def test_bit_metrics_perfect(self):
        y=np.arange(16)
        m=real_sdr_metrics(y,y)
        self.assertEqual(m["ber"],0.0)
        self.assertEqual(m["ser"],0.0)


if __name__=="__main__":
    unittest.main()
