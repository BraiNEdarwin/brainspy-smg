import unittest
from bspysmg.utils import inputs
import copy


class Test_Inputs(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Inputs, self).__init__(*args, **kwargs)
        self.configs = {}
        self.configs["driver"] = {
            "instrument_type": "cdaq_to_cdaq",
            "inverted_output": True,
            "instruments_setup": {
                "multiple_devices": False,
                "average_io_point_difference": True,
                "activation_sampling_frequency": 1000,
                "readout_sampling_frequency": 10000,
                "trigger_source": "cDAQ1/segment1",
                "activation_instrument": "cDAQ1Mod3",
                "activation_channels": [0,2,5,3,4,6,1],
                "activation_voltage_ranges":
                [[-0.55,0.325],[-0.95,0.55],[-1,0.6],[-1,0.6],[-1,0.6],[-0.95,0.55],[-0.55,0.325]],
                "activation_channel_mask": [1,1,1,1,1,1,1],
                "readout_instrument": "cDAQ1Mod4",
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
            "offset": [-0.15,-0.25,-0.25,-0.25,-0.25,-0.25,-0.15]
        }


    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            configs, func = inputs.get_input_generator()
    

    def test_normal_inputs(self):
        CONFIGS = copy.deepcopy(self.configs)
        try:
            configs, func = inputs.get_input_generator(CONFIGS)
        except:
            self.fail("Exeution Failed")


    def test_configs_keys(self):
        CONFIGS = copy.deepcopy(self.configs)

        with self.assertRaises(KeyError):
            configs_1 = copy.deepcopy(self.configs)
            del configs_1['driver']['instruments_setup']
            configs, func = inputs.get_input_generator(configs_1)

        with self.assertRaises(KeyError):
            configs_2 = copy.deepcopy(self.configs)
            del configs_2['driver']['instruments_setup']['activation_sampling_frequency']
            configs, func = inputs.get_input_generator(configs_2)
        
        with self.assertRaises(KeyError):
            configs_3 = copy.deepcopy(self.configs)
            del configs_3["input_data"]['phase']
            configs, func = inputs.get_input_generator(configs_3)

        with self.assertRaises(KeyError):
            configs_4 = copy.deepcopy(self.configs)
            del configs_4["input_data"]['amplitude']
            configs, func = inputs.get_input_generator(configs_4)

        with self.assertRaises(KeyError):
            configs_5 = copy.deepcopy(self.configs)
            del configs_5["input_data"]['offset']
            configs, func = inputs.get_input_generator(configs_5)


if __name__ == '__main__':
    unittest.main()