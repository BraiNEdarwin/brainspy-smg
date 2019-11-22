from bspyproc.processors.processor_mgr import get_hardware_processor
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
        self.processor = get_hardware_processor(configs["processor"])
        self.generate_inputs = get_input_generator(configs)
        # define internal attributes
        self.end_batch = configs["ramp_points"] + configs["batch_points"]
        self.nr_points_ramped_signal = configs["batch_points"] + 2 * configs["ramp_points"]
        self.filter_ramp = np.zeros(self.nr_points_ramped_signal, type=bool)
        self.index_start_filter = configs["ramp_points"]
        self.index_finish_filter = self.end_batch
        self.filter_ramp[self.index_start_batch:self.index_finish_batch] = True

    def get_batch(self, input_batch):
        # Ramp input batch
        batch_ramped = self.ramp_input_batch(input_batch)
        # Readout output signal
        outputs_ramped = self.processor.get_output(batch_ramped)
        return outputs_ramped[self.filter_ramp]

    def ramp_input_batch(self, input_batch):
        input_batch_ramped = np.zeros((self.config["number_inputs"], self.nr_points_ramped_signal))
        for j in range(self.config["number_inputs"]):
            input_batch_ramped[j, 0:self.config["ramp_points"]] = np.linspace(0, input_batch[j, 0], self.config["ramp_points"])
            input_batch_ramped[j, self.config["ramp_points"]: self.end_batch] = input_batch[j, :]
            input_batch_ramped[j, self.end_batch:] = np.linspace(input_batch[j, -1], 0, self.config["ramp_points"])
        return input_batch_ramped

    def get_data(self):

        # Initialize output containers
        outputs = np.zeros((self.configs["batch_points"], self.configs["nr_outputs"]))
        # Generate inputs
        inputs = self.generate_inputs(self.configs["inputs"])

        start_batch = time.time()
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

        # Initialize output containers
        outputs = np.zeros((self.configs["nr_batches"], self.configs["batch_points"], self.configs["nr_outputs"]))
        # Generate inputs
        time_points = np.arange(batch * configs['sample_points'], (batch + 1) * configs['sample_points'])

        for batch in range(configs["batches"]):
            start_batch = time.time()
            outputs[batch, :] = self.get_batch(input_batch)
            if batch % 10 == 0:  # Save after every 10 batches
                self.save_data(inputs, outputs)
            end_batch = time.time()
            print('outputs collection for part ' + str(batch + 1) + ' of ' + str(configs["batches"]) + ' took ' + str(end_batch - start_batch) + ' sec.')
        return inputs, outputs, self.configs
