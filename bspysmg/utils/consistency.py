import os
import time
import numpy as np
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs
from bspysmg.measurement.data.output.sampler_mgr import Sampler
from brainspy.utils.io import create_directory_timestamp

class ConsistencyChecker(Sampler):

    def __init__(self, main_dir, repetitions=1, sampler_configs_name='sampler_configs.json', reference_batch_name='reference_batch.npz', charging_signal_name='charging_signal.npz'):
        super().__init__(load_configs(os.path.join(main_dir, sampler_configs_name)))
        _, self.batch_size, _ = self.init_configs()
        path_to_file = os.path.join(main_dir, reference_batch_name)
        with np.load(path_to_file) as data:
            self.reference_outputs = data['outputs']
            self.reference_inputs = data['inputs'].T
        path_to_file = os.path.join(main_dir, charging_signal_name)
        with np.load(path_to_file) as data:
            self.chargingup_outputs = data['outputs']
            self.chargingup_inputs = data['inputs'].T
        self.nr_samples = len(self.reference_outputs)
        assert self.nr_samples % self.batch_size == 0, f"Batch length {self.batch_size} is not a multiple of the reference signal length {self.nr_samples}; possible data missmatch!"
        self.results_dir = create_directory_timestamp(main_dir, 'consistency_check')
        self.results_filename = os.path.join(self.results_dir, 'consistency_results.npz')
        self.repetitions = repetitions

    def get_data(self):
        results = np.zeros((self.repetitions,) + self.reference_outputs.shape)
        deviations = np.zeros(self.repetitions)
        correlation = np.zeros(self.repetitions)
        deviation_chargeup = []
        for batch, batch_indices in enumerate(self.batch_generator(len(self.chargingup_outputs), self.batch_size)):
            # Generate inputs (without ramping)
            inputs = self.chargingup_inputs[:, batch_indices]
            # Get outputs (without ramping)
            outputs = self.get_batch(inputs)
            charging_signal_deviations = np.sqrt(np.mean((outputs - self.chargingup_outputs[batch_indices])**2))
            deviation_chargeup.append(charging_signal_deviations)
            print(f'* Charging up: Batch {batch+1}/{int(len(self.chargingup_outputs)/self.batch_size)}\n RMSE deviation of batch from original data: {charging_signal_deviations:.2f} (nA)\n')
        print('\n')
        # Initialize sampling loop
        for trial in range(self.repetitions):
            start_trial = time.time()
            for batch, batch_indices in enumerate(self.batch_generator(self.nr_samples, self.batch_size)):

                # Generate inputs (without ramping)
                inputs = self.reference_inputs[:, batch_indices]
                # Get outputs (without ramping; raming is done in the get_batch method)
                outputs = self.get_batch(inputs)
                results[trial, batch_indices] = outputs
                # if (batch % 1) == 0:
                #     self.plot_waves(inputs.T, outputs, batch)
            end_trial = time.time()
            deviations[trial] = np.sqrt(np.mean((results[trial] - self.reference_outputs)**2))
            correlation[trial] = np.corrcoef(results[trial].T, self.reference_outputs.T)[0, 1]
            print(f'* Consistency check {trial+1}/{self.repetitions} took {end_trial - start_trial :.2f} sec. with {batch+1} batches')
            print(f"\tCorr: {correlation[trial]:.2f} ; Deviation: {deviations[trial]:.2f}")
        self.close_processor()

        np.savez(self.results_filename, results=results, deviations=deviations, correlation=correlation)
        print(f'Data saved in { self.results_filename}')
        return results, deviations, correlation, deviation_chargeup


def consistency_check(main_dir, repetitions=1, sampler_configs_name='sampler_configs.json', reference_batch_name='reference_batch.npz', charging_signal_name='charging_signal.npz'):

    sampler = ConsistencyChecker(main_dir, repetitions=repetitions, sampler_configs_name=sampler_configs_name, reference_batch_name=reference_batch_name, charging_signal_name=charging_signal_name)

    outputs, deviations, correlation, deviation_chargeup = sampler.get_data()

    mean_output = np.mean(outputs, axis=0)
    std_output = np.std(outputs, axis=0)

    plt.figure()
    plt.plot(sampler.reference_outputs, 'r', label='reference signal', alpha=0.5)
    plt.plot(mean_output, 'k', label='mean over repetitions', alpha=0.5)
    plt.plot(mean_output + std_output, ':k', alpha=0.5)
    plt.plot(mean_output - std_output, ':k', label='stdev over repetitions', alpha=0.5)

    plt.title(f'Consistency over {sampler.repetitions} trials with same input')
    plt.legend()
    plt.savefig(os.path.join(sampler.results_dir, 'consistency_check'))

    plt.figure()
    plt.plot(mean_output - sampler.reference_outputs, "b", label="mean - reference")
    plt.plot(std_output, ":k", label="stdev over repetitions")
    plt.plot(-std_output, ":k")
    plt.title("Difference Mean Signal and Reference Signal (nA)")
    plt.legend()
    plt.savefig(os.path.join(sampler.results_dir, 'diff_mean-ref'))

    plt.figure()
    plt.hist(deviations)
    plt.title("RMSE Deviations of Reference Signal (nA)")
    plt.savefig(os.path.join(sampler.results_dir, 'hist_deviations'))

    plt.figure()
    plt.plot(deviation_chargeup)
    plt.title("RMSE deviations (nA) while charging up")
    plt.savefig(os.path.join(sampler.results_dir, 'deviations_while_charging_up'))

    plt.show()


if __name__ == '__main__':
    consistency_check('tmp/data/training/TEST/Brains_testing_2021_05_07_142853', repetitions=1)
