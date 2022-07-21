import unittest
from bspysmg.utils.iv import simple
import numpy as np
from bspysmg import TEST_MODE
from brainspy.utils.io import load_configs

class Test_Simple(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Simple, self).__init__(*args, **kwargs)
        import os 
        print(os.getcwd())
        self.configs = load_configs("configs/utils/brains_ivcurve_template_simple.yaml")

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):

        try:
            measurement = simple.IVMeasurement(self.configs)
        except Exception:
            self.fail("Failed in creating error hist")


if __name__ == '__main__':
    import bspysmg

    bspysmg.TEST_MODE = 'HARDWARE_CDAQ'
    unittest.main()
