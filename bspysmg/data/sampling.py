"""
File containing a class for sampling a device.
"""
from __future__ import generator_stop
from brainspy.utils.manager import get_driver
from bspysmg.utils.inputs import get_input_generator
from brainspy.utils.io import create_directory_timestamp as mkdir
from brainspy.utils.io import save_configs
from bspysmg.utils.plots import plot_waves
from bspysmg.utils.inputs import get_random_phase
from more_itertools import grouper
import numpy as np
import time
import os
from typing import Tuple, Generator


class Sampler:
    def __init__(self, configs: dict) -> None:
        """
        Initialises the driver from which the data will be acquired, and it stores the sampling
        configurations internally. It also calculates the amplitude and vertical offset of the input
        waveform from the control voltage ranges of the  driver, if the 'amplitude' and 'offset'
        keys are not present in the configuration dictionary.

        Parameters
        ----------
        configs : dict
            Sampling configurations, the dictionary has the following keys:
            1. save_directory: str
            Directory where the all the sampling data will be stored.

            2. data_name: str
            Inside the path specified on the variable save_directory, a folder will be created,
            with the format: <data_name>+<current_timestamp>. This variable specified the
            prefix of that folder before the timestamp.

            3. driver: dict
            Dictionary containing the driver configurations. For more information check the
            documentation about this configuration file, check the documentation of
            brainspy.processors.hardware.drivers.ni.setup.NationalInstrumentsSetup

            4. input_data : dict
            Dictionary containing the information necessary to create the input sampling data.
            4.1 input_distribution: str
            It determines the wave shape of the input. Two main options availeble 'sawtooth'
            and 'sine'. The first option will create saw-like signals, and the second
            sine-wave signals. Sawtooth signals have more coverage on the edges of the
            input range.

            4.2 activation_electrode_no: int
            Number of activation electrodes in the device that wants to be sampled.

            4.3 readout_electrode_no : int
            Number of readout electrodes in the device that wants to be sampled.

            4.4 input_frequency: list
            Base frequencies of the input waves that will be created. In order to optimise
            coverage, irrational numbers are recommended. The list should have the same
            length as the activation electrode number. E.g., for 7 activation electrodes:
            input_frequency = [2, 3, 5, 7, 13, 17, 19]

            4.5 random_phase_shift_each : int
            The input data for each activation electrode can be shifted horizontally by
            changing its phase. This is a randomised process. This variable represents
            at how many batched samples the phase will be randomised. This will cover the
            input space faster and increase the data quality such that less data is needed.
            If the value is 0, no phase shift will be applied.

            4.6 factor : float
            Given factor by which the input frequencies will be multiplied after square
            rooting them.

            4.7 amplitude : Optional[list[float]]
            Amplitude of the generated input wave signal. It is calculated according to the
            minimum and maximum ranges of each electrode. Where the amplitude value should
            correspond with (max_range_value - min_range_value) / 2. If no amplitude is
            given it will be automatically calculated from the driver configurations for
            activation electrode ranges. If it wants to be manually set, the offset
            variable should also be included in the dictionary.

            4.8 offset: Optional[list[float]]
            Vertical offset of the generated input wave signal. It is calculated according
            to the minimum and maximum ranges of each electrode. Where the offset value
            should correspond with (max_range_value + min_range_value) / 2. If no offset
            is given it will be automatically calculated from the driver configurations for
            activation electrode ranges. If it wants to be manually set, the offset
            variable should also be included in the dictionary.

            4.9 ramp_time: float
            Time that will be taken before sending each batch to go from zero to the first
            point of the batch and to zero from the last point of the batch.

            4.10  batch_time:
            Time that the sampling of each batch will take.

            4.11 number_batches: int
            Number of batches that will be sampled. A default value of 3880 is reccommended.

            4.12 randomise_phase_each: int (Optional)
            Specifies at how many epochs the phases will be randomly generated again. 
        """
        self.driver = get_driver(configs["driver"])
        self.configs = configs
        if 'amplitude' not in self.configs[
                'input_data'] and 'offset' not in self.configs['input_data']:
            self.config_offset_and_amplitude()

    def config_offset_and_amplitude(self) -> None:
        """
        It extracts the offset and amplitude values that the input waveforms will have, according
        to the voltage ranges specified in the driver. It stores them into the configuration
        dictionary (input_data/amplitude and input_data/offset).
        """
        assert isinstance(
            self.configs['driver']['instruments_setup']
            ['activation_voltage_ranges'],
            list), "Voltage ranges should be passed as a list"
        assert self.configs['driver']['instruments_setup'][
            'activation_voltage_ranges'] != [], "Empty array for voltage ranges"

        voltage_ranges = np.array(self.configs['driver']['instruments_setup']
                                  ['activation_voltage_ranges'])
        amplitude = (voltage_ranges[:, 1] - voltage_ranges[:, 0]) / 2
        offset = (voltage_ranges[:, 0] + voltage_ranges[:, 1]) / 2
        self.configs['input_data']['amplitude'] = amplitude
        self.configs['input_data']['offset'] = offset
        print(f"Amplitude: {str(amplitude)}")
        print(f"Offset: {str(offset)}")

    def sample_batch(self, x: np.array) -> np.array:
        """
        Perform a sampling operation over one input batch. This method includes
        a ramping from zero until the beginning of the batch until the first point, and
        a ramping from the last point to zero. The information from ramps is filtered
        before it is returned.

        Parameters
        ----------
        x : np.array
            Input batch that will be sent to the device in order to sample its corresponding output.
            The dimension of the sample should be (activation_electrode_no, batch_size).

        Returns
        -------
        output : np.array
            Readout of the device when applying the input batch. The size of the output will be
            (batch_size, readout_electrode_no).
        """
        # Ramp input batch (0.5 sec up and down)
        ramped_input = self.ramp_input(x)
        outputs_ramped = self.driver.forward_numpy(ramped_input.T)
        return outputs_ramped[self.filter_ramp]

    def ramp_input(self, x: np.array):
        """
        The input batch is prepared for sampling on the device by ramping it from zero until the
        beginning of the batch until the first point, and a ramping from the last point to zero.

        Parameters
        ----------
        x : np.array
            Input batch that will be sent to the device in order to sample its corresponding output.
            The dimension of the sample should be (activation_electrode_no, batch_size). 

        Returns
        -------
        ramped_input : np.array
            Input batch that will be sent to the device in order to sample its corresponding output
            with an additional ramping from zero until the beginning of the batch until the first
            point, and a ramping from the last point to zero.
        """
        dimensions = self.configs["input_data"]["activation_electrode_no"]
        ramp = int(self.configs["input_data"]["ramp_points"])
        ramped_input = np.zeros((dimensions, self.nr_points_ramped_signal))
        for j in range(dimensions):
            ramped_input[j, 0:ramp] = np.linspace(0, x[j, 0], ramp)
            ramped_input[j, ramp:self.end_batch] = x[j, :]
            ramped_input[j, self.end_batch:] = np.linspace(x[j, -1], 0, ramp)
        return ramped_input

    def sample(self, plot_interval: int = 1) -> str:
        """
        Performs a full sampling operation, divided into several batches,
        according to the configurations given to the class. It stores each
        batch on a txt file. Additionally it can also show the inputs and
        corresponding outputs of a batch in a saved plot.

        Parameters
        ----------
        plot_interval : Optional[int]
            It sets after how many batches it will save a plot of the current batch,
            that shows the inputs and outputs. By default 1 (on every batch).

        Returns
        -------
        save_directory : str
            Folder where all the sampling data is stored.
        """
        # Create a directory and file for saving
        self.save_batch()
        # Initialize configs
        total_number_samples, batch_size, input_dict = self.init_configs()

        # Initialize sampling loop
        all_time_points = np.arange(
            total_number_samples) / input_dict["sampling_frequency"]
        if 'phase' in self.configs['input_data']:
            phase = np.array(self.configs['input_data']['phase'])
            if len(phase.shape) == 1:
                phase = phase[:,np.newaxis]
        else:
            phase = np.zeros(
                (self.configs["input_data"]["activation_electrode_no"], 1))
        phase_randomisation_count = 0
        for batch, batch_indices in enumerate(
                self.get_batch_indices(total_number_samples, batch_size)):
            start_batch = time.time()

            # Generate inputs (without ramping)
            batch += 1
            time_points = all_time_points[batch_indices]
            inputs = self.generate_inputs(time_points,
                                          input_dict['input_frequency_numpy'], phase,
                                          input_dict['amplitude_numpy'],
                                          input_dict['offset_numpy'])
            # Get outputs (without ramping)
            outputs = self.sample_batch(inputs)
            self.save_batch(inputs.T, outputs)
            end_batch = time.time()
            if (batch % plot_interval) == 0:
                input_no = self.configs["input_data"][
                    "activation_electrode_no"]
                output_no = self.configs["input_data"]["readout_electrode_no"]
                legend = self.get_header(input_no, output_no).split(',')
                plot_waves(inputs.T, outputs, input_no, output_no, batch,
                           legend, self.configs["save_directory"])
            print(
                f"Outputs collection for batch {batch} of {input_dict['number_batches']} "
                + f"took {end_batch - start_batch} sec. Estimated time left: "+convert(batch, input_dict['number_batches'], (end_batch - start_batch)) )
            phase_randomisation_count += 1
            if input_dict[
                    'random_phase_shift_each'] >= phase_randomisation_count:
                phase = get_random_phase(
                    self.configs["input_data"]["activation_electrode_no"])
                phase_randomisation_count = 0
        self.close_driver()
        return self.configs["save_directory"]

    def get_batch_indices(self, sample_no: int,
                          batch_size: int) -> Generator[int, None, None]:
        """
        Collects data length into indices and yields them into fixed-length chunks or blocks.

        Parameters
        ----------
        sample_no : int
            Total number of samples to be sent to the device.
        batch_size : int
            Desired block size in which the total number of sizes will be divided.

        Yields
        -------
        indices : list[int]
            List of indices corresponding to the generated input signal.

        """
        print('Start batching...')
        batches = grouper(np.arange(sample_no), batch_size)
        while True:
            try:
                indices = list(next(batches))
                if None in indices:
                    indices = [index for index in indices if index is not None]
                yield indices
            except StopIteration:
                return

    def get_header(self, input_no: int, output_no: int) -> str:
        """
        Gets the header of the txt file data, so that is stored as a string format.

        Parameters
        ----------
        input_no : int
            The input electrode number.
        output_no : int
            The output electrode number.

        Returns
        -------
        str
            The headers of each input and output batch in a string format.
        """
        header = ""
        for i in range(input_no):
            header += f"Input {i}, "
        for i in range(output_no):
            if i < (output_no - 1):
                header += f"Output {i}, "
            else:
                header += f"Output {i}"
        return header

    def init_configs(self) -> Tuple[int, int, dict]:
        """
        Initializes the configurations for performing sampling operation. It initializes
        and returns the number of samples, the batch size and an input dictionary for
        sampling.

        Returns
        -------
        tuple
            total_number_samples: int
                The total number of samples to be generated in the input dataset.
            batch_size: int
                The batch size to be used for processing the input dataset in batches.
            input_dict: dict
                The configurations for sampling operation with following keys:
                    - activation_electrode_no: int
                        Number of activation electrodes in the device that wants to be sampled.
                    - readout_electrode_no : int
                        Number of readout electrodes in the device that wants to be sampled.
                    - input_frequency: list
                        Base frequencies of the input waves that will be created. In order to optimise
                        coverage, irrational numbers are recommended. The list should have the same
                        length as the activation electrode number. E.g., for 7 activation electrodes:
                        input_frequency = [2, 3, 5, 7, 13, 17, 19]
                    - phase : float
                        Horizontal shift of the input signals. It is recommended to have random numbers
                        which are different for the training, validation and test datasets. These
                        numbers will be square rooted and multiplied by a given factor.
                    - amplitude : Optional[list[float]]
                        Amplitude of the generated input wave signal. It is calculated according to the
                        minimum and maximum ranges of each electrode. Where the amplitude value should
                        correspond with (max_range_value - min_range_value) / 2. If no amplitude is
                        given it will be automatically calculated from the driver configurations for
                        activation electrode ranges. If it wants to be manually set, the offset
                        variable should also be included in the dictionary.
                    - offset: Optional[list[float]]
                        Vertical offset of the generated input wave signal. It is calculated according
                        to the minimum and maximum ranges of each electrode. Where the offset value
                        should correspond with (max_range_value + min_range_value) / 2. If no offset
                        is given it will be automatically calculated from the driver configurations for
                        activation electrode ranges. If it wants to be manually set, the offset
                        variable should also be included in the dictionary.
                    - number_batches: int
                        Number of batches that will be sampled. A default value of 3880 is reccommended.
        """
        input_dict, self.generate_inputs = get_input_generator(self.configs)

        batch_size = input_dict["sampling_frequency"] * input_dict["batch_time"]
        total_number_samples = input_dict["number_batches"] * batch_size

        # define internal attributes
        self.end_batch = int(input_dict["ramp_points"] +
                             input_dict["batch_points"])
        self.nr_points_ramped_signal = int(input_dict["batch_points"] +
                                           2 * input_dict["ramp_points"])

        self.filter_ramp = np.zeros(self.nr_points_ramped_signal, dtype=bool)
        self.filter_ramp[int(input_dict["ramp_points"]):int(self.end_batch
                                                            )] = True

        return int(total_number_samples), int(batch_size), input_dict

    def save_batch(self, *args) -> None:
        """
        Stores a batch of data on the same IO.dat text comma separated values file.
        
        Parameters
        ----------
        *args: Tuple[np.array]
            A set of input and output arrays of a batch which is to be saved.
        """
        if len(args) > 0:
            with open(self.path_to_iodata, '+a') as f:
                data = np.column_stack(args)
                np.savetxt(f, data)
                f.flush()
        else:
            print(f'Saving in {self.configs["save_directory"]}')
            path_to_file = mkdir(self.configs["save_directory"],
                                 self.configs["data_name"])
            self.configs["save_directory"] = path_to_file
            save_configs(self.configs,
                         os.path.join(path_to_file, 'sampler_configs.json'))
            header = self.get_header(
                self.configs["input_data"]["activation_electrode_no"],
                self.configs["input_data"]["readout_electrode_no"])
            self.path_to_iodata = path_to_file + "/IO.dat"
            with open(self.path_to_iodata, 'wb') as f:
                np.savetxt(f, [], header=header)

    def close_driver(self) -> None:
        """
        Adequately closes the connection to the drivers.
        """
        self.driver.close_tasks()
        print('Instrument task closed')

def convert(num_batches, total_batches, time_taken):
    """
    Converts the number of batches and the last time taken for measuring them into
    a string containing an estimation of the time left in HH:MM:SS format.

    Parameters
    ----------
    num_batches: int
        Number of batches that have been already sampled.
    total_batches:int
        Total batches that will be sampled.
    time_taken: float
        Time that the last measurement has taken.
    
    Return
    ----------
        str: A string containing the time left in HH:MM:SS
    """
    seconds = (total_batches - num_batches) * time_taken
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)
