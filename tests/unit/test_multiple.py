import unittest
from bspysmg.utils.iv import multiple
import numpy as np
from bspysmg import TEST_MODE
from brainspy.utils.io import load_configs
from bspysmg.utils import plots

class Test_Multiple(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Multiple, self).__init__(*args, **kwargs)
        self.configs = load_configs("configs/utils/brains_ivcurve_template.yaml")

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):
        try:
            self.configs['driver']['instruments_setup']['A']['activation_channel_mask'] = [1,1,1,1,1,1,1]
            measurement = multiple.MultiIVMeasurement(self.configs)
            configs, inputs, output = measurement.run_test(close_driver=False)
            self.configs['input_signal']['input_signal_type'] = "sine"
            configs, inputs, output = measurement.run_test(close_driver=True)
            self.configs['driver']['instruments_setup']['A']['activation_channel_mask'] = [1,1,0,1,1,1,1]
            measurement = multiple.MultiIVMeasurement(self.configs)
            configs, inputs, output = measurement.run_test(close_driver=True)
            plots.multi_iv_plot(configs, inputs, output, show_plot=False)
        except Exception:
            self.fail("Failed in creating error hist")


if __name__ == '__main__':
    import bspysmg

    bspysmg.TEST_MODE = 'HARDWARE_CDAQ'
    unittest.main()
