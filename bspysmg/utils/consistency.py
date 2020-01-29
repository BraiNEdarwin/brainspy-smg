import os
import time
import torch
import numpy as np
from bspyalgo.utils.io import load_configs
from bspyproc.processors.processor_mgr import get_processor
from bspysmg.measurement.data.output.sampler_mgr import Sampler


class ConsistencyChecker(Sampler):

    def __init__(self, configs):
        super().__init__(configs)
        self.init_configs()
        self.path_to_file = self.configs['path_to_file']
        self.hw_processor = self.processor
        self.model_processor = get_processor(self.get_configs(configs['processor_model']))
        self.total_batch_number = 2

    def get_configs(self, model_path):
        configs = {}
        configs["platform"] = "simulation"
        configs["architecture"] = "single_device"
        configs["simulation_type"] = "neural_network"
        configs["network_type"] = "nn_model"
        configs["torch_model_dict"] = model_path
        return configs

    def get_data(self, reference_batch, repetition_no=1, use_model=False, results_filename='consistency_results'):
        if use_model:
            self.processor = self.model_processor
        else:
            self.processor = self.hw_processor

        results_filename = os.path.join(self.path_to_file, results_filename)

        batch = 0
        batch_size = int(len(reference_batch['inputs']) / self.total_batch_number)

        start = 0
        current_batch = int(batch_size)

        iteration_no = int(len(reference_batch['inputs']) / batch_size)

        results = np.zeros(shape=(repetition_no, reference_batch['outputs'].shape[0]))
        error_results = np.zeros(shape=(repetition_no))
        # model_result = np.zeros(shape=(repetition_no, reference_batch['output_data'].shape[0], reference_batch['output_data'].shape[1]))

        # Initialize sampling loop
        for i in range(repetition_no):
            for j in range(iteration_no):
                start_batch = time.time()

                # Generate inputs (without ramping)
                batch += 1
                input_data = reference_batch['inputs'][start:current_batch].T

                # Get outputs (without ramping)
                results[i][start:current_batch] = self.get_batch(input_data)[:, 0]
                end_batch = time.time()
                start += batch_size
                current_batch += batch_size
                print(f'Outputs collection for batch {str(batch)} of {self.total_batch_number} took {str(end_batch - start_batch)} sec.')
            error = reference_batch['outputs'] - results[i]
            error_results[i] = np.mean(error ** 2)

        self.close_processor()
        np.savez(results_filename, results=results, error=error_results)
        print(f'Data saved in {results_filename}.npz')
        return results, error_results


def consistency_check(data_path, repetition_no=100, reference_batch_name='reference_batch.npz', hardware_configs_name='sampler_configs.json', model_name='trained_network.pt'):

    reference_batch_path = os.path.join(data_path, reference_batch_name)
    hw_processor_configs_path = os.path.join(data_path, hardware_configs_name)
    model_path = os.path.join(data_path, model_name)

    reference_batch = np.load(reference_batch_path)
    assert len(reference_batch['inputs']) == len(reference_batch['outputs'])

    configs = load_configs(hw_processor_configs_path)
    configs['path_to_file'] = data_path
    configs['processor_model'] = model_path

    sampler = ConsistencyChecker(configs)

    hw_results = sampler.get_data(reference_batch, repetition_no=repetition_no)
    model_results = sampler.get_data(reference_batch, repetition_no=1, use_model=True)


if __name__ == '__main__':

    consistency_check('tmp/input/model_afterny')
