import unittest
from bspysmg.utils import inputs
import copy
import numpy as np


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


    def test_sine_wave(self):
        with self.assertRaises(TypeError):
            sine_wave = inputs.sine_wave()

        try:
            sine_wave = inputs.sine_wave(np.array(range(0)), 0.1, 5, 0.1, 10)
            sine_wave = inputs.sine_wave(np.array(range(100)), 60, 0, 5, 0)
            sine_wave = inputs.sine_wave(np.array(range(100)), 0.1, 0, 5, 0)
            sine_wave = inputs.sine_wave(np.array(range(100)), 0.1, 5, 5, 0)
            sine_wave = inputs.sine_wave(np.array(range(100)), 0.1, 5, 0.1, 0)
            sine_wave = inputs.sine_wave(np.array(range(100)), 0.1, 5, 0.1, 10)
            assert sine_wave.flatten().shape[0] == 100
        except:
            self.fail("Failed creating sine_wave")

        with self.assertRaises(TypeError):
            sine_wave = inputs.sine_wave(list(range(100)), 60, 0, 5, 0)

        with self.assertRaises(TypeError):
            sine_wave = inputs.sine_wave(100, 60, 0, 5, 0)

        with self.assertRaises(TypeError):
            sine_wave = inputs.sine_wave(np.array(range(100)), "60", 0, 5, 0)

        with self.assertRaises(TypeError):
            sine_wave = inputs.sine_wave(np.array(range(100)), 60, "5", 5, 0)

        with self.assertRaises(TypeError):
            sine_wave = inputs.sine_wave(np.array(range(100)), 60, 0, "5", 0)

        with self.assertRaises(TypeError):
            sine_wave = inputs.sine_wave(np.array(range(100)), 60, 0, 5, "10")


    def test_sawtooth_wave(self):
        with self.assertRaises(TypeError):
            sawtooth_wave = inputs.sawtooth_wave()

        try:
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 60, 0, 5, 0)
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 0.1, 0, 5, 0)
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 0.1, 5, 5, 0)
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 0.1, 5, 0.1, 0)
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 0.1, 5, 0.1, 10)
            assert sawtooth_wave.flatten().shape[0] == 100
        except:
            self.fail("Failed creating sawtooth_wave")

        with self.assertRaises(TypeError):
            sawtooth_wave = inputs.sawtooth_wave(list(range(100)), 60, 0, 5, 0)

        with self.assertRaises(TypeError):
            sawtooth_wave = inputs.sawtooth_wave(100, 60, 0, 5, 0)

        with self.assertRaises(TypeError):
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), "60", 0, 5, 0)

        with self.assertRaises(TypeError):
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 60, "5", 5, 0)

        with self.assertRaises(TypeError):
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 60, 0, "5", 0)

        with self.assertRaises(TypeError):
            sawtooth_wave = inputs.sawtooth_wave(np.array(range(100)), 60, 0, 5, "10")


    def test_generate_sinewave(self):
        with self.assertRaises(TypeError):
            sinewave = inputs.generate_sinewave()

        try:
            sinewave = inputs.generate_sinewave(100, 60, 5, 0)
            sinewave = inputs.generate_sinewave(20, 0.1, 5, 0)
            sinewave = inputs.generate_sinewave(100, 0.1, 5, 5)
            sinewave = inputs.generate_sinewave(100, 0.1, 5, 0.1)
            assert sinewave.flatten().shape[0] == 100
        except:
            self.fail("Failed creating generate_sinewave")

        with self.assertRaises(ZeroDivisionError):
            sinewave = inputs.generate_sinewave(0, 60, 0, 5)

        with self.assertRaises(ValueError):
            sinewave = inputs.generate_sinewave(-10, 60, 0, 5)

        with self.assertRaises(TypeError):
            sinewave = inputs.generate_sinewave(40.0, 60, 0, 5)

        with self.assertRaises(TypeError):
            sinewave = inputs.generate_sinewave(np.array(range(100)), 60, 0, 5)

        with self.assertRaises(TypeError):
            sinewave = inputs.generate_sinewave(100, "60", 0, 5)

        with self.assertRaises(TypeError):
            sinewave = inputs.generate_sinewave(100, 60, "5", 5)

        with self.assertRaises(TypeError):
            sinewave = inputs.generate_sinewave(100, 60, 0, "5")


    def test_generate_sawtooth_simple(self):
        with self.assertRaises(TypeError):
            sawtooth_simple = inputs.generate_sawtooth_simple()

        try:
            sawtooth_simple = inputs.generate_sawtooth_simple(-5, 5, 100, False)
            sawtooth_simple = inputs.generate_sawtooth_simple(-5, 5, 100, True)
            sawtooth_simple = inputs.generate_sawtooth_simple(-6.5, 7.5, 10)
            assert sawtooth_simple.flatten().shape[0] == 10
        except:
            self.fail("Failed creating generate_sinewave")

        with self.assertRaises(ValueError):
            sawtooth_simple = inputs.generate_sawtooth_simple(5, 15, 20)

        with self.assertRaises(AssertionError):
            sawtooth_simple = inputs.generate_sawtooth_simple(5, -5, 20)

        with self.assertRaises(AssertionError):
            sawtooth_simple = inputs.generate_sawtooth_simple(-5, 5, 5)


if __name__ == '__main__':
    unittest.main()