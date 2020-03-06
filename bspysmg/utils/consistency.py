import os
import time
import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.utils.io import load_configs
from bspysmg.measurement.data.output.sampler_mgr import Sampler


class ConsistencyChecker(Sampler):

    def __init__(self, configs):
        super().__init__(load_configs(configs['path_to_sampler_configs']))
        _, batch_size, _ = self.init_configs()
        self.configs_checker = configs
        path_to_file = os.path.join(self.configs_checker['path_to_reference_data'], self.configs_checker['reference_batch_name'])
        with np.load(path_to_file) as data:
            self.reference_outputs = data['outputs']
            self.reference_inputs = data['inputs'].T
        path_to_file = os.path.join(self.configs_checker['path_to_reference_data'], self.configs_checker['data_name'])
        with np.load(path_to_file) as data:
            self.pre_reference_outputs = data['outputs'][-90000:]
            self.pre_reference_inputs = data['inputs'][-90000:].T
        self.nr_samples = len(self.reference_outputs)
        assert self.nr_samples % batch_size == 0, f"Batch length {batch_size} is not a multiple of the reference signal length {self.nr_samples}; possible data missmatch!"
        self.configs_checker['batch_size'] = batch_size

        self.results_filename = os.path.join(self.configs_checker['path_to_reference_data'], 'consistency_results.npz')

    def get_data(self):

        results = np.zeros((self.configs_checker['repetitions'],) + self.reference_outputs.shape)
        deviations = np.zeros(self.configs_checker['repetitions'])
        correlation = np.zeros(self.configs_checker['repetitions'])
        deviation_chargeup = []
        for batch, batch_indices in enumerate(self.batch_generator(len(self.pre_reference_outputs), self.configs_checker['batch_size'])):
            # Generate inputs (without ramping)
            inputs = self.pre_reference_inputs[:, batch_indices]
            # Get outputs (without ramping)
            outputs = self.get_batch(inputs)
            presignal_deviations = np.sqrt(np.mean((outputs - self.pre_reference_outputs[batch_indices])**2))
            deviation_chargeup.append(presignal_deviations)
            print(f'Charging up: Deviation of batch {batch+1} from pre-reference signal {presignal_deviations}')

        # Initialize sampling loop
        for trial in range(self.configs_checker['repetitions']):
            start_trial = time.time()
            for batch, batch_indices in enumerate(self.batch_generator(self.nr_samples, self.configs_checker['batch_size'])):

                # Generate inputs (without ramping)
                inputs = self.reference_inputs[:, batch_indices]
                # Get outputs (without ramping)
                outputs = self.get_batch(inputs)
                results[trial, batch_indices] = outputs
                # if (batch % 1) == 0:
                #     self.plot_waves(inputs.T, outputs, batch)
            end_trial = time.time()
            deviations[trial] = np.sqrt(np.mean((results[trial] - self.reference_outputs)**2))
            correlation[trial] = np.corrcoef(results[trial].T,self.reference_outputs.T)[0,1]
            print(f'Consistency check {trial+1}/{self.configs_checker["repetitions"]} took {end_trial - start_trial :.2f} sec. with {batch+1} batches')
            print(f"Corr: {correlation[trial]:.2f} ; Deviation: {deviations[trial]:.2f}")
        self.close_processor()

        np.savez(self.results_filename, results=results, deviations=deviations,correlation=correlation)
        print(f'Data saved in { self.results_filename}')
        return results, deviations, correlation,deviation_chargeup


def consistency_check(configs_path):

    configs = load_configs(configs_path)
    sampler = ConsistencyChecker(configs)

    outputs, deviations, correlation, deviation_chargeup = sampler.get_data()

    mean_output = np.mean(outputs, axis=0)
    std_output = np.std(outputs, axis=0)
    
    plt.figure()
    plt.plot(mean_output, 'k', label='mean over repetitions')
    plt.plot(mean_output + std_output, ':k')
    plt.plot(mean_output - std_output, ':k', label='stdev over repetitions')
    plt.plot(sampler.reference_outputs, 'r', label='reference signal')
    plt.title(f'Consistency over {sampler.configs_checker["repetitions"]} trials with same input')
    plt.legend()
    plt.savefig(os.path.join(configs["path_to_reference_data"] ,'consistency_check'))

    plt.figure()
    plt.plot(mean_output-sampler.reference_outputs,"b",label="mean - reference")
    plt.plot(std_output,":k",label="stdev over repetitions")
    plt.plot(-std_output,":k")
    plt.title("Difference Mean Signal and Reference Signal")
    plt.legend()
    plt.savefig(os.path.join(configs["path_to_reference_data"] ,'diff_mean-ref'))

    plt.figure()
    plt.hist(deviations)
    plt.title("Deviations of Reference Signal")
    plt.savefig(os.path.join(configs["path_to_reference_data"] ,'hist_deviations'))

    plt.figure()
    plt.plot(deviation_chargeup)
    plt.title("DEVIATIONS WHILE CHARGING UP")
    plt.savefig(os.path.join(configs["path_to_reference_data"] ,'deviations_while_charging_up'))

    plt.show()
    print("DONE!")


if __name__ == '__main__':

    consistency_check('configs/consistency_check_configs.json')
    
