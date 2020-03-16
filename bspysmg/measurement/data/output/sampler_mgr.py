# -*- coding: utf-8 -*-
"""
Script to sample a device using waves
@author: HC Ruiz
"""
from __future__ import generator_stop
from bspyproc.processors.processor_mgr import get_processor
from bspysmg.measurement.data.input.input_mgr import get_input_generator
from bspyalgo.utils.io import create_directory_timestamp as mkdir
from bspyalgo.utils.io import save_configs
from more_itertools import grouper
import matplotlib.pyplot as plt
import numpy as np
import time
import os


class Sampler:

    def __init__(self, configs):
        configs["processor"]['waveform'] = {'slope_lengths': configs["input_data"]['ramp_time'] * configs["processor"]['sampling_frequency']}  # add this because needed in setup_mgr.py of processors
        self.configs = configs
        # define processor and input generator
        self.processor = get_processor(configs["processor"])

    def get_batch(self, input_batch):
        # Ramp input batch (0.5 sec up and down)
        batch_ramped = self.ramp_input_batch(input_batch)
        # Readout output signal
        outputs_ramped = self.processor.get_output(batch_ramped.T)
        return outputs_ramped[self.filter_ramp]

    def ramp_input_batch(self, input_batch):
        '''Ramps the input up and down for ech batch
        @author: M. Boon
        '''
        # TODO: can we do without ramping up and down to zero?
        dimensions = self.configs["input_data"]["input_electrodes"]
        ramp = int(self.configs["input_data"]["ramp_points"])
        input_batch_ramped = np.zeros((dimensions, self.nr_points_ramped_signal))
        for j in range(dimensions):
            input_batch_ramped[j, 0:ramp] = np.linspace(0, input_batch[j, 0], ramp)
            input_batch_ramped[j, ramp: self.end_batch] = input_batch[j, :]
            input_batch_ramped[j, self.end_batch:] = np.linspace(input_batch[j, -1], 0, ramp)
        return input_batch_ramped

    def get_data(self):
        # Create a directory and file for saving
        self.save_data()
        # Initialize configs
        total_number_samples, batch_size, input_dict = self.init_configs()

        # Initialize sampling loop
        all_time_points = np.arange(total_number_samples) / input_dict["sampling_frequency"]
        for batch, batch_indices in enumerate(self.batch_generator(total_number_samples, batch_size)):
            start_batch = time.time()
            # Generate inputs (without ramping)
            batch += 1
            time_points = all_time_points[batch_indices]
            inputs = self.generate_inputs(time_points, input_dict['input_frequency'],
                                          input_dict['phase'], input_dict['amplitude'],
                                          input_dict['offset'])
            # Get outputs (without ramping)
            outputs = self.get_batch(inputs)
            self.save_data(inputs.T, outputs)
            end_batch = time.time()
            if (batch % 1) == 0:
                self.plot_waves(inputs.T, outputs, batch)
            print(f'Outputs collection for batch {str(batch)} of {str(input_dict["number_batches"])} took {str(end_batch - start_batch)} sec.')
        self.close_processor()
        return self.configs["save_directory"]

    def batch_generator(self, nr_samples, batch):
        print('Start batching...')
        batches = grouper(np.arange(nr_samples), batch)
        while True:
            try:
                indices = list(next(batches))
                if None in indices:
                    indices = [index for index in indices if index is not None]
                yield indices
            except StopIteration:
                return

    def get_header(self, input_nr, output_nr):
        header = ""
        for i in range(input_nr):
            header += f"Input {i}, "
        for i in range(output_nr):
            if i < (output_nr - 1):
                header += f"Output {i}, "
            else:
                header += f"Output {i}"
        return header

    def init_configs(self):
        input_dict, self.generate_inputs = get_input_generator(self.configs)
        total_number_samples = input_dict["number_batches"] * input_dict["sampling_frequency"] * input_dict["batch_time"]
        batch_size = int(input_dict["sampling_frequency"] * input_dict["batch_time"])
        # define internal attributes
        self.end_batch = int(input_dict["ramp_points"] + input_dict["batch_points"])
        self.nr_points_ramped_signal = int(input_dict["batch_points"] + 2 * input_dict["ramp_points"])
        self.filter_ramp = np.zeros(self.nr_points_ramped_signal, dtype=bool)
        self.filter_ramp[int(input_dict["ramp_points"]):int(self.end_batch)] = True
        return total_number_samples, batch_size, input_dict

    def save_data(self, *args):
        if len(args) > 0:
            with open(self.path_to_iodata, '+a') as f:
                data = np.column_stack(args)
                np.savetxt(f, data)
                f.flush()
        else:
            print(f'Saving in {self.configs["save_directory"]}')
            path_to_file = mkdir(self.configs["save_directory"], self.configs["data_name"])
            self.configs["save_directory"] = path_to_file
            save_configs(self.configs, os.path.join(path_to_file, 'sampler_configs.json'))
            header = self.get_header(self.configs["input_data"]["input_electrodes"],
                                     self.configs["input_data"]["output_electrodes"])
            self.path_to_iodata = path_to_file + "/IO.dat"
            with open(self.path_to_iodata, 'wb') as f:
                np.savetxt(f, [], header=header)

    def load_data(self, path):
        data = np.loadtxt(path)
        inputs = data[:, :self.configs["input_data"]["input_electrodes"]]
        outputs = data[:, -self.configs["input_data"]["output_electrodes"]:]
        return inputs, outputs, self.configs

    def plot_waves(self, inputs, outputs, batch):
        nr_inputs = self.configs["input_data"]["input_electrodes"]
        nr_outputs = self.configs["input_data"]["output_electrodes"]
        legend = self.get_header(nr_inputs, nr_outputs).split(',')
        plt.figure()
        plt.suptitle(f'Data for NN training in batch {batch}')
        plt.subplot(211)
        plt.plot(inputs)
        plt.ylabel('Inputs (V)')
        plt.xlabel('Time points (a.u.)')
        plt.legend(legend[:nr_inputs])
        plt.subplot(212)
        plt.plot(outputs)
        plt.ylabel('Outputs (nA)')
        plt.legend(legend[-nr_outputs:])
        plt.xlabel('Time points (a.u.)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.configs["save_directory"], 'example_batch'))
        plt.close()

    def close_processor(self):
        """
        Experiments in hardware require that the connection with the drivers is closed.
        This method helps closing this connection when necessary.
        """
        try:
            self.processor.close_tasks()
            print('Instrument task closed')
        except AttributeError:
            print('There is no closing function for the current processor configuration. Skipping.')


class Repeater(Sampler):

    def __init__(self, configs):
        super().__init__(configs)

    def batch_generator(self, nr_samples, batch):
        print('Repeating the experiment...')
        count = 0
        while nr_samples > count * batch:
            indices = list(range(batch))
            if None in indices:
                indices = [index for index in indices if index is not None]
            yield indices
            count += 1


if __name__ == '__main__':

    from bspyalgo.utils.io import load_configs
    import matplotlib.pyplot as plt

    CONFIGS = load_configs('configs/sampling/sampling_configs_template.json')
    sampler = Sampler(CONFIGS)
    # CONFIGS = load_configs('configs/sampling/toy_sampling_configs_template.json')
    # sampler = Sampler(CONFIGS)
    # CONFIGS = load_configs('configs/sampling/toy_sampling_configs_template.json')
    # sampler = Repeater(CONFIGS)

    path_to_data = sampler.get_data()

    INPUTS, OUTPUTS, INFO_DICT = sampler.load_data(path_to_data)

    # OUTPUTS = OUTPUTS.reshape((-1,INFO_DICT['input_data']["batch_points"])).T
    plt.figure()
    plt.hist(OUTPUTS, 500)
    plt.show()
