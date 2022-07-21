import unittest
from bspysmg.utils.iv import simple
import numpy as np
from bspysmg import TEST_MODE
from brainspy.utils.io import load_configs

class Test_Simple(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Simple, self).__init__(*args, **kwargs)
        self.configs = load_configs("configs/utils/brains_ivcurve_template_simple.yaml")

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):

        try:
            measurement = simple.IVMeasurement(self.configs)
            measurement.iv_curve(
             measurement.driver.get_voltage_ranges()[0, 0].item(),
             measurement.driver.get_voltage_ranges()[0, 1].item(),
             point_no=10,
             input_electrode=0,
             show_plot=True,
             close=False)
            measurement.iv_curve(
             measurement.driver.get_voltage_ranges()[1, 0].item(),
             measurement.driver.get_voltage_ranges()[1, 1].item(),
             point_no=10,
             input_electrode=1,
             show_plot=True,
             close=True)
        except Exception:
            self.fail("Failed in creating error hist")


if __name__ == '__main__':
    import bspysmg

    bspysmg.TEST_MODE = 'HARDWARE_CDAQ'
    unittest.main()
