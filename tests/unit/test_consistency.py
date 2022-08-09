import unittest
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils
from bspysmg.utils import consistency
from bspysmg import TEST_MODE
import numpy as np

class Test_Consistency(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Consistency, self).__init__(*args, **kwargs)
        configs = {}
        configs = {}
        configs['processor_type'] = 'simulation'
        configs["electrode_effects"] = {}
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 1 #10
        configs["waveform"]["slope_length"] = 0 #30
        self.device_configs = configs

        model_data = {}
        model_data["info"] = {}
        model_data["info"]["model_structure"] = {
            "hidden_sizes": [90, 90, 90],
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
        }
        model_data["info"]['electrode_info'] = {}
        model_data["info"]['electrode_info']['electrode_no'] = 8
        model_data["info"]['electrode_info']['activation_electrodes'] = {}
        model_data["info"]['electrode_info']['activation_electrodes']['electrode_no'] = 7
        model_data["info"]['electrode_info']['activation_electrodes'][
                'voltage_ranges'] = np.array([[-0.55, 0.325], [-0.95, 0.55],
                                            [-1., 0.6], [-1., 0.6], [-1., 0.6],
                                            [-0.95, 0.55], [-0.55, 0.325]])
        model_data["info"]['electrode_info']['output_electrodes'] = {}
        model_data["info"]['electrode_info']['output_electrodes']['electrode_no'] = 1
        model_data["info"]['electrode_info']['output_electrodes']['amplification'] = [28.5]
        model_data["info"]['electrode_info']['output_electrodes']['clipping_value'] = None
        self.model_data = model_data

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ"
                         or TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_consistency(self):
        try:
            consistency.consistency_check(
                    'tests/data/',
                    repetitions=1,
                    charge_device=True,
                    show_plots=False)
        except Exception:
            self.fail("Not possible to run consistency check")

    @unittest.skipUnless(TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_consistency_with_model(self):
        model = Processor(self.device_configs, self.model_data['info'])
        model = TorchUtils.format(model)
        try:
            consistency.consistency_check('tests/data',
                    repetitions=1,
                    charge_device=True,
                    show_plots=False,
                    model=model)
        except Exception:
            self.fail("Not possible to run consistency check")



if __name__ == '__main__':
    unittest.main()