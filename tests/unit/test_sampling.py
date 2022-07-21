import unittest
from bspysmg import TEST_MODE
from bspysmg.data import sampling
from bspysmg.data.postprocess import post_process
import copy


class Test_Sampling(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Sampling, self).__init__(*args, **kwargs)
        self.configs = {}
        self.configs["driver"] = {
            "instrument_type": "cdaq_to_cdaq",
            "inverted_output": True,
            "instruments_setup": {
                "multiple_devices":
                False,
                "average_io_point_difference":
                True,
                "activation_sampling_frequency":
                1000,
                "readout_sampling_frequency":
                10000,
                "trigger_source":
                "cDAQ1/segment1",
                "activation_instrument":
                "cDAQ1Mod3",
                "activation_channels": [0, 2, 5, 3, 4, 6, 1],
                "activation_voltage_ranges": [[-0.55, 0.325], [-0.95, 0.55],
                                              [-1, 0.6], [-1, 0.6], [-1, 0.6],
                                              [-0.95, 0.55], [-0.55, 0.325]],
                "activation_channel_mask": [1, 1, 1, 1, 1, 1, 1],
                "readout_instrument":
                "cDAQ1Mod4",
                "readout_channels": [0]
            },
            "amplification": [39.5],
            "gain_info": "50MOhm"
        }
        self.configs["input_data"] = {
            "input_distribution": "sawtooth",
            "activation_electrode_no": 7,
            "readout_electrode_no": 1,
            "input_frequency": [2, 3, 5, 7, 13, 17, 19],
            "phase": [-0.57, 0.25, -1.54, 2.17, 0.08, 0.15, -0.65],
            "ramp_time": 0.5,
            "batch_time": 0.5,
            "number_batches": 1,
            "amplitude": [0.55, 0.95, 0.95, 0.95, 0.95, 0.95, 0.55],
            "offset": [-0.15, -0.25, -0.25, -0.25, -0.25, -0.25, -0.15]
        }

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            sampling.Sampler()

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_normal_inputs(self):
        CONFIGS = copy.deepcopy(self.configs)
        try:
            sampling.Sampler(CONFIGS)
        except Exception:
            self.fail("Exeution Failed")

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_configs_keys(self):

        with self.assertRaises(AssertionError):
            configs_1 = copy.deepcopy(self.configs)
            del configs_1['driver']['instruments_setup']
            sampling.Sampler(configs_1)

        with self.assertRaises(KeyError):
            configs_2 = copy.deepcopy(self.configs)
            del configs_2['driver']['instruments_setup'][
                'activation_voltage_ranges']
            sampling.Sampler(configs_2)

        with self.assertRaises(AssertionError):
            configs_3 = copy.deepcopy(self.configs)
            configs_3['driver']['instruments_setup'][
                'activation_voltage_ranges'] = 1
            sampling.Sampler(configs_3)

        with self.assertRaises(AssertionError):
            configs_4 = copy.deepcopy(self.configs)
            configs_4['driver']['instruments_setup'][
                'activation_voltage_ranges'] = []
            sampling.Sampler(configs_4)


if __name__ == '__main__':
    unittest.main()