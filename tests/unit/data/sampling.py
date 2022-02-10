from typing import Type
import unittest
from bspysmg.data import sampling
from brainspy.utils.io import load_configs
from bspysmg.data.postprocess import post_process

class Test_Sampling(unittest.TestCase):

    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            sampler = sampling.Sampler()
    """
    def test_normal_inputs(self):
        CONFIGS = load_configs('./configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')
        try:
            sampler = sampling.Sampler(CONFIGS)
        except:
            self.fail("Exeution Failed")
    """
    def test_configs_keys(self):

        with self.assertRaises(KeyError):
            configs_1 = load_configs('./configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')
            del configs_1['driver']['instruments_setup']
            sampler = sampling.Sampler(configs_1)

        with self.assertRaises(KeyError):
            configs_2 = load_configs('./configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')
            del configs_2['driver']['instruments_setup']['activation_voltage_ranges']
            sampler = sampling.Sampler(configs_2)

        with self.assertRaises(AssertionError):
            configs_3 = load_configs('./configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')
            configs_3['driver']['instruments_setup']['activation_voltage_ranges'] = 1
            sampler = sampling.Sampler(configs_3)

        with self.assertRaises(AssertionError):
            configs_4 = load_configs('./configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')
            configs_4['driver']['instruments_setup']['activation_voltage_ranges'] = []
            sampler = sampling.Sampler(configs_4)

if __name__ == '__main__':
    unittest.main()