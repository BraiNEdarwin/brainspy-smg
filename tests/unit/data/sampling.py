from typing import Type
import unittest
from bspysmg.data import sampling
from brainspy.utils.io import load_configs
from bspysmg.data.postprocess import post_process

class Test_Sampling(unittest.TestCase):

    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            sampler = sampling.Sampler()
    
    def test_normal_inputs(self):
        CONFIGS = load_configs('./configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')
        try:
            sampler = sampling.Sampler(CONFIGS)
        except:
            self.fail("Exeution Failed")


if __name__ == '__main__':
    unittest.main()