"""
File containing a class for measuring IV curves on a single device (or surrogate model).
"""
import numpy as np
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.manager import get_driver
from brainspy.utils.transforms import linear_transform
from bspysmg.utils.inputs import generate_sawtooth_simple, generate_sinewave


class MultiIVMeasurement():
    def __init__(self, configs: dict, save_plot=None, show_plot=True) -> None:
        """
        Initializes the drivers for which IV curve is to be plotted. It uses a config dict to
        initialize the driver. The drivers can be the DNPU device itself (on chip training) or
        a surrogate model (off chip training). This class allows for the measurement of IV curves
        for several devices in a single PCB.

        Parameters
        ----------
            configs : dict
                Dictionary containing the configurations for IV measurements with
                following keys:

                - input_signal: dict
                    The configuration of signal with following keys:
                    - input_signal_type: str
                        The type of signal to generate - sawtooth or sine.
                    - time_in_seconds: int
                        The length of the signal to generate in seconds.
                - devices: list
                    List of devices for which IV response is to be computed. This list contains the
                    names of all the devices (A,B,C,D etc) involved in the experiment.
        """
        self.configs = configs
        self.input_signal = self.configs['input_signal']
        self.driver = get_driver(self.configs['driver'])

    def run_test(
            self,
            experiments=["IV1", "IV2", "IV3", 
                         "IV4", "IV5", "IV6",
                         "IV7"], 
            close_driver: bool =True) -> None:
        """
        Generates the IV response of devices to a sawtooth or sine wave and shows it
        on the screen. It uses configs dictionary with the following keys:

        - devices: list
            List of devices for which IV response is to be computed. This list contains the
            names of all the devices (A,B,C,D etc) involved in the experiment.
        - driver: dict
            It contains the configurations for each device in the experiment which
            are defined in the devices list.
        - close_driver: boolean
            Whether to close the driver or not after running the IV curve.
        """
        # save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)

        self.devices_in_experiments = {}
        output = {}
        inputs = {}
        output_array = []
        input_arrays = []
        self.index_prog = {}
        self.index_prog["all"] = 0
        for dev in self.configs['devices']:
            self.index_prog[dev] = 0
        for i, exp in enumerate(experiments):
            output[exp] = {}
            inputs[exp] = {}
            self.devices_in_experiments[exp] = self.configs['devices'].copy()
            output_array = self.driver.forward_numpy(
                self.create_input_arrays(inputs[exp]))

            for j, dev in enumerate(self.configs['devices']):
                output[exp][dev] = output_array[:, j]
        self.index_prog["all"] = 0
        if close_driver:
            self.driver.close_tasks()
        
        return self.configs, inputs, output

    def create_input_arrays(self, inputs_dict: dict) -> np.array:
        """
        Generates input signal arrays for each device in inputs_dict dictionary that will
        be used to measure the IV response of those devices. The devices can be the DNPU
        device or a surrogate model. It uses configs dictionary with the following keys:

        - devices: list
            List of devices for which IV response is to be computed. This list contains the
            names of all the devices (A,B,C,D etc) involved in the experiment.
        - shape: int
            The length of the generated signal.
        - driver: dict
            It contains the configurations for each device in the experiment which
            are defined in the devices list.

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
        Generates input signal to compute the IV response of DNPU device or
        a surrogate model. It uses configs dictionary with the following keys:

        - input_signal_type: str
            The type of signal to generate - sawtooth or sine.
        - shape: int
            The length of the generated signal.
        - direction: str ['up'/'down']
            The Direction of the sawtooth. If true, the sawtooth will go first up
            and then down. If False, the sawtooth will go first down and then up.
            By default up.
        - frequency: int
            The frequency of the sine wave signal.

        Parameters
        ----------
            input_range : float
                Maximum voltage that the signal will achieve. Minimum voltage is 0.

        Returns
        ----------
            result : np.array
                Generated sawtooth or sine signal.
        """
        if self.input_signal['input_signal_type'] == 'sawtooth':
            input_data = generate_sawtooth_simple(
                input_range[0], input_range[1], self.configs['shape'],
                self.input_signal['direction'])
        else:
            input_data = generate_sinewave(
                self.configs['shape'],
                self.configs['driver']['instruments_setup']['activation_sampling_frequency'],
                amplitude=1)  # Max from the input range
            
            input_data = linear_transform(input_range[0], input_range[1], -1, 1, input_data )
            input_data[-1] = 0
            input_data[0] = 0
        return input_data