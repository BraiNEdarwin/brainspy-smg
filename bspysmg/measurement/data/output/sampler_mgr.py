from bspyproc.processors.processor_mgr import get_processor
from bspysmg.measurement.data.input.input_mgr import get_input_generator
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
        size_batch = input_dict["sampling_frequency"] * input_dict["batch_time"]
        size_input = (input_dict["number_batches"], input_dict["input_electrodes"], size_batch)
        inputs = np.zeros(size_input)
        size_output = (input_dict["number_batches"], size_batch, 1)
        outputs = np.zeros(size_output)

        # Generate inputs (without ramping)
        for batch in range(input_dict["number_batches"]):
            start_batch = time.time()
            inputs[batch, :] = self.generate_inputs(input_dict)
            outputs[batch, :] = self.get_batch(inputs[batch, :])
            if batch % 10 == 0:  # Save every 10 batches
                self.save_data(inputs, outputs)
            end_batch = time.time()
            print('outputs collection for part ' + str(batch + 1) + ' of ' + str(configs["batches"]) + ' took ' + str(end_batch - start_batch) + ' sec.')
        return inputs, outputs, self.configs


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs

    CONFIGS = load_configs('configs/sampling/toy_sampling_configs_template.json')
    sampler = BatchSampler(CONFIGS)
    RAW_DATA = sampler.get_data()
