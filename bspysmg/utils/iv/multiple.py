import numpy as np
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver
from bspysmg.utils.plots import multi_iv_plot
from bspysmg.utils.inputs import generate_sawtooth_simple, generate_sinewave


class MultiIVMeasurement():
    def __init__(self, configs: dict) -> None:
        """
        Initializes the configurations for measuring the IV curves of several devices.

        Parameters
        ----------
            configs : dict
                Dictionary containing the configurations for IV measurements with
                following keys:

                - input_signal: str
                    The type of signal to generate - sawtooth or sine.
                - devices: list
                    List of devices for which IV response is to be computed.
        """
        self.configs = configs
        self.input_signal = self.configs['input_signal']
        self.index_prog = {}
        self.index_prog["all"] = 0
        for dev in self.configs['devices']:
            self.index_prog[dev] = 0

    def run_test(self) -> None:
        """
        Generates the IV response of devices to a sawtooth or sine wave and plots it
        on the screen.
        """
        # save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)

        self.driver = get_driver(self.configs['driver'])
        experiments = ["IV1", "IV2", "IV3", "IV4", "IV5", "IV6", "IV7"]
        self.devices_in_experiments = {}
        output = {}
        inputs = {}
        output_array = []
        input_arrays = []
        for i, exp in enumerate(experiments):
            output[exp] = {}
            inputs[exp] = {}
            self.devices_in_experiments[exp] = self.configs['devices'].copy()
            output_array = self.driver.forward_numpy(
                self.create_input_arrays(inputs[exp]))

            for j, dev in enumerate(self.configs['devices']):
                output[exp][dev] = output_array[:, j]
        self.driver.close_tasks()
        multi_iv_plot(configs, inputs, output)

    def create_input_arrays(self, inputs_dict: dict) -> np.array:
        """
        Generates input signal arrays for each device in inputs_dict dictionary that will
        be used to measure the IV response of those devices. The devices can be the DNPU
        device or a surrogate model. 

        Parameters
        ----------
            inputs_dict : dict
                Dictionary containing the devices for which IV curve is to be measured
                as keys.
        
        Returns
        ----------
            inputs_array : np.array
                Generated signal arrays for each device.
        """
        #inputs_dict = {}
        inputs_array = []

        for dev in self.configs['devices']:

            inputs_dict[dev] = np.zeros(
                (self.configs["driver"]['instruments_setup'][dev]
                 ['activation_channel_mask'].count(1), self.configs['shape']
                 ))  # creates a zeros array for each '1' in the mask entry

            if self.configs["driver"]['instruments_setup'][dev][
                    'activation_channel_mask'][self.index_prog["all"]] == 1:
                inputs_dict[dev][
                    self.index_prog[dev], :] = self.gen_input_wfrm(
                        self.configs["driver"]['instruments_setup'][dev]
                        ['activation_voltage_ranges'][self.index_prog["all"]])
                self.index_prog[dev] += 1

            else:
                self.devices_in_experiments["IV" + str(self.index_prog["all"] +
                                                       1)].remove(dev)

            inputs_array.extend(inputs_dict[dev])

        inputs_array = np.array(inputs_array)
        self.index_prog["all"] += 1

        return inputs_array.T

    def gen_input_wfrm(self, input_range: float) -> np.array:
        """
        Generates multiple input signals to compute the IV response of DNPU device or
        a surrogate model. It uses configs dictionary key input_signal_type to
        generate sawtooth or sine signal.

        Parameters
        ----------
            input_range : float
                Maximum voltage that the signal will achieve.

        Returns
        ----------
            result : np.array
                Generated signals. 
        """
        if self.input_signal['input_signal_type'] == 'sawtooth':
            input_data = generate_sawtooth_simple(
                input_range[0], input_range[1], self.configs['shape'],
                self.input_signal['direction'])
        elif self.input_signal['input_signal_type'] == 'sine':
            input_data = generate_sinewave(
                self.configs['shape'],
                self.configs["driver"]['sampling_frequency'],
                input_range[1])  # Max from the input range
            input_data[-1] = 0
        else:
            print("Specify input_signal type")

        return input_data


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    configs = {}
    configs['results_base_dir'] = 'tmp/tests/iv'
    configs['show_plots'] = True
    configs['devices'] = ["A"]
    # configs['devices'] = [
    #     "A", "B", "C", "D", "E"
    # ]  #["D"]  # To remove devices from this list, set the mask to zero first in the configs.
    configs['shape'] = 500  # length of the experiment
    configs['input_signal'] = {}
    #configs['input_signal']['voltage_range'] = [1.2, 0.5]
    configs['input_signal'][
        'input_signal_type'] = 'sawtooth'  # Type of signal to be created in the input. It can either be 'sine' or 'sawtooth'
    configs['input_signal'][
        'time_in_seconds'] = 5  # time_in_seconds in seconds
    configs['input_signal']['direction'] = 'up'

    # @TODO: Create templates for other setups
    configs['driver'] = load_configs(
        'configs/utils/brains_ivcurve_template.yaml')

    test = MultiIVMeasurement(configs)
    test.run_test()
    #suite = unittest.TestSuite()
    #suite.addTest(IVtest(configs))
    #unittest.TextTestRunner().run(suite)
