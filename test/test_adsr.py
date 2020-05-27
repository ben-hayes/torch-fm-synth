import unittest

import torch

from torch_fm_synth.adsr import ADSR

class TestADSR(unittest.TestCase):
    def test_can_create(self):
        adsr = ADSR()
        self.assertIsInstance(adsr, ADSR)

    def test_is_torch_nn_module(self):
        adsr = ADSR()
        self.assertIsInstance(adsr, torch.nn.Module)