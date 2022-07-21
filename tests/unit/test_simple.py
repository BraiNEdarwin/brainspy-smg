import unittest
from bspysmg.utils.iv import simple
import numpy as np
from bspysmg import TEST_MODE


class Test_Simple(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Simple, self).__init__(*args, **kwargs)
        self.configs = {
            'results_base_dir': 'temp/test/multpleIV',
            'show_plots': True,
            'devices': ["A"],
            'shape': 1000,
        }
        self.configs["input_signal"] = {
            'input_signal_type': 'sawtooth',
            'time_in_seconds': 5,
            'direction': 'up'
        }
        self.configs["driver"] = {
            'instrument_type': 'cdaq_to_cdaq',
            'amplification': 39.5,
            'inverted_output': True
        }
        self.configs["driver"]["instruments_setup"] = {
            'multiple_devices':
            False,
            'trigger_source':
            'cDAQ1/segment1',
            'activation_sampling_frequency':
            200,
            'readout_sampling_frequency':
            10000,
            'average_io_point_difference':
            False,
            'activation_instrument':
            'cDAQ1Mod3',
            'activation_channels': [6, 0, 1, 5, 2, 4, 3],
            'activation_voltage_ranges':
            [[-0.9, 0.8], [-0.7, 1.0], [-0.9, 1.1], [-0.8, 0.8], [-0.8, 0.9],
             [-0.6, 0.5], [-1, 1.3]],
            'activation_channel_mask': [1, 1, 1, 1, 1, 1, 1],
            'readout_instrument':
            'cDAQ1Mod4',
            'readout_channels': [2]
        }

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):

        try:
            measurement = simple.IVMeasurement(self.configs)
        except Exception:
            self.fail("Failed in creating error hist")


if __name__ == '__main__':
    unittest.main()
