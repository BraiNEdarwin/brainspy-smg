import unittest
import numpy as np
from bspysmg import TEST_MODE
from bspysmg.data import sampling
from bspysmg.data.postprocess import post_process
from brainspy.utils.io import load_configs
import copy


class Test_Sampling(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Sampling, self).__init__(*args, **kwargs)
        self.configs = load_configs('configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            sampler = sampling.Sampler()
            sampler.close_driver()

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_normal_inputs(self):
        CONFIGS = copy.deepcopy(self.configs)
        CONFIGS['input_data']['number_batches'] = 2
        CONFIGS['input_data']['batch_time'] = 0.1
        CONFIGS['save_directory'] = 'tests/data/sampling'
        if 'phase' not in CONFIGS['input_data']:
            CONFIGS['input_data']['phase'] = np.zeros((self.configs["input_data"]["activation_electrode_no"])).tolist()
        try:
            sampler = sampling.Sampler(CONFIGS)
            sampler.sample()
            sampler.close_driver()
        except Exception:
            self.fail("Exeution Failed")
        #sampler.close_driver()

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_normal_inputs_no_phase(self):
        CONFIGS = copy.deepcopy(self.configs)
        CONFIGS['input_data']['number_batches'] = 2
        CONFIGS['input_data']['batch_time'] = 0.1
        CONFIGS['save_directory'] = 'tests/data/sampling'
        if 'phase' in CONFIGS['input_data']:
            del CONFIGS['input_data']['phase']
        try:
            sampler = sampling.Sampler(CONFIGS)
            sampler.sample()
            sampler.close_driver()
        except Exception:
            self.fail("Exeution Failed")
        #sampler.close_driver()

    # @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
    #                      or TEST_MODE == "HARDWARE_NIDAQ",
    #                      "Hardware test is skipped for simulation setup.")
    # def test_configs_keys(self):

    #     with self.assertRaises(AssertionError):
    #         configs_1 = copy.deepcopy(self.configs)
    #         del configs_1['driver']['instruments_setup']
    #         sampler = sampling.Sampler(configs_1)
    #         sampler.close_driver()

    #     with self.assertRaises(AssertionError) :
    #         configs_2 = copy.deepcopy(self.configs)
    #         del configs_2['driver']['instruments_setup'][
    #             'activation_voltage_ranges']
    #         sampler = sampling.Sampler(configs_2)
    #         sampler.close_driver()

    #     with self.assertRaises(AssertionError):
    #         configs_3 = copy.deepcopy(self.configs)
    #         configs_3['driver']['instruments_setup'][
    #             'activation_voltage_ranges'] = 1
    #         sampler = sampling.Sampler(configs_3)
    #         sampler.close_driver()

    #     with self.assertRaises(AssertionError):
    #         configs_4 = copy.deepcopy(self.configs)
    #         configs_4['driver']['instruments_setup'][
    #             'activation_voltage_ranges'] = []
    #         sampler = sampling.Sampler(configs_4)
    #         sampler.close_driver()

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_get_header(self):
        configs = copy.deepcopy(self.configs)
        sampler = sampling.Sampler(configs)
        in_length = 7
        out_length = 7
        header = sampler.get_header(in_length, out_length)
        split_header = header.split(',')
        self.assertEqual(len(split_header) ,(in_length + out_length))
        sampler.close_driver()
if __name__ == '__main__':
    unittest.main()