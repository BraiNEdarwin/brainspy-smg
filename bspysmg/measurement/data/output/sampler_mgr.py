from bspyproc.processors.processor_mgr import get_processor
from bspysmg.measurement.data.input.input_mgr import get_input_generator
from more_itertools import grouper
import numpy as np
import time


def get_sampler(configs):
    if configs["batch_type"] == "batches":
        return BatchSampler(configs)
    elif configs["batch_type"] == "single_batch":
        return Sampler(configs)
    else:
        raise NotImplementedError(f"Batch type {configs['batch_type']} not recognised!")


class Sampler:

    def __init__(self, configs):
        self.configs = configs
        # define processor and input generator
        self.processor = get_processor(configs["processor"])
        self.configs["input_data"], self.generate_inputs = get_input_generator(configs["input_data"])
        # define internal attributes
        self.end_batch = int(configs["input_data"]["ramp_points"] + configs["input_data"]["batch_points"])
        self.nr_points_ramped_signal = int(configs["input_data"]["batch_points"] + 2 * configs["input_data"]["ramp_points"])
        self.filter_ramp = np.zeros(self.nr_points_ramped_signal, dtype=bool)
        self.index_start_filter = int(configs["input_data"]["ramp_points"])
        self.index_finish_filter = int(self.end_batch)
        self.filter_ramp[self.index_start_filter:self.index_finish_filter] = True

    def get_batch(self, input_batch):
        # Ramp input batch (0.5 sec up and down)
        batch_ramped = self.ramp_input_batch(input_batch)
        # Readout output signal
        outputs_ramped = self.processor.get_output(batch_ramped.T)
        return outputs_ramped[self.filter_ramp]

    def ramp_input_batch(self, input_batch):
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
        start_batch = time.time()
        # Generate inputs (without rammping)
        inputs = self.generate_inputs(self.configs["input_data"])
        outputs = self.get_batch(inputs)
        self.save_data(inputs, outputs)
        end_batch = time.time()
        print(f'Outputs collection took {str(end_batch - start_batch)} sec.')
        return inputs, outputs, self.configs

    def save_data(self, inputs, outputs):
        print(f'Saving in {self.configs["save_directory"]}')
        # TODO: implement saving as method of sampler: save inputs,outputs AND configs
    #                        output=outputs * configs['amplification'] / configs['postgain'],
    #                        freq=configs['freq'],
    #                        sampleTime=configs['sample_time'],
    #                        sample_frequency=configs['sample_frequency'],
    #                        phase=configs['phase'],
    #                        amplitude=configs['amplitude'],
    #                        offset=configs['offset'],
    #                        amplification=configs['amplification'],
    #                        electrodeSetup=configs['electrode_setup'],
    #                        gain_info=configs['gain_info'],
    #                        filename='training_nn_data'


class BatchSampler(Sampler):

    def __init__(self, configs):
        super().__init__(configs)

    def get_data(self):
        # Initialize data containers
        input_dict = self.configs["input_data"]
        total_number_samples = input_dict["number_batches"] * input_dict["sampling_frequency"] * input_dict["batch_time"]
        length_batch = int(input_dict["sampling_frequency"] * input_dict["batch_time"])
        size_input = (input_dict["input_electrodes"], total_number_samples)
        inputs = np.zeros(size_input)
        size_output = (total_number_samples, input_dict["output_electrodes"])
        outputs = np.zeros(size_output)

        all_time_points = np.arange(total_number_samples) / input_dict["sampling_frequency"]
        for batch, batch_indices in enumerate(self.batch_generator(total_number_samples, length_batch)):
            start_batch = time.time()
            # Generate inputs (without ramping)
            batch += 1
            time_points = all_time_points[batch_indices]
            inputs[:, batch_indices] = self.generate_inputs(time_points, input_dict['input_frequency'],
                                                            input_dict['phase'], input_dict['amplitude'],
                                                            input_dict['offset'])
            # Get outputs (without ramping)
            outputs[batch_indices, :] = self.get_batch(inputs[:, batch_indices])
            if batch % 10 == 0:  # Save every 10 batches
                self.save_data(inputs, outputs)
            end_batch = time.time()
            print(f'Outputs collection for batch {str(batch)} of {str(input_dict["number_batches"])} took {str(end_batch - start_batch)} sec.')
        return inputs, outputs, self.configs

    def batch_generator(self, nr_samples, batch):
        print('Start batching...')
        batches = grouper(np.arange(nr_samples), batch)
        while True:
            indices = list(next(batches))
            if None in indices:
                indices = [index for index in indices if index is not None]
            yield indices


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs

    CONFIGS = load_configs('configs/sampling/toy_sampling_configs_template.json')
    sampler = BatchSampler(CONFIGS)
    RAW_DATA = sampler.get_data()
