import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs
from bspysmg.data.sampling import Sampler
from brainspy.utils.io import create_directory_timestamp
from brainspy.utils.pytorch import TorchUtils


class ConsistencyChecker(Sampler):
    def __init__(self,
                 main_dir,
                 repetitions=1,
                 sampler_configs_name='sampler_configs.json',
                 reference_batch_name='reference_batch.npz',
                 charging_signal_name='charging_signal.npz',
                 model=None):
        super(ConsistencyChecker, self).__init__(
            load_configs(os.path.join(main_dir, sampler_configs_name)))
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
        self.results_dir = create_directory_timestamp(main_dir,
                                                      'consistency_check')
        self.results_filename = os.path.join(self.results_dir,
                                             'consistency_results.npz')
        self.repetitions = repetitions
        self.model = model

    def get_data(self):
        results = np.zeros((self.repetitions, ) + self.reference_outputs.shape)
        deviations = np.zeros(self.repetitions)
        correlation = np.zeros(self.repetitions)
        deviation_chargeup = []
        if self.model is not None:
            model_results = results.copy()
            model_deviations = deviations.copy()
            model_correlation = correlation.copy()
            model_deviation_chargeup = []

        for batch, batch_indices in enumerate(
                self.get_batch_indices(len(self.chargingup_outputs),
                                       self.batch_size)):
            # Generate inputs (without ramping)
            inputs = self.chargingup_inputs[:, batch_indices]
            # Get outputs (without ramping)
            outputs = self.sample_batch(inputs)
            charging_signal_deviations = np.sqrt(
                np.mean((outputs - self.chargingup_outputs[batch_indices])**2))
            deviation_chargeup.append(charging_signal_deviations)
            print(
                f'\n* Charging up: Batch {batch+1}/{int(len(self.chargingup_outputs)/self.batch_size)}\n\tRMSE deviation of measured device output against original device output: {charging_signal_deviations:.2f} (nA)'
            )
            if self.model is not None:
                model_outputs = self.get_model_batch(inputs)
                model_charging_signal_deviations = np.sqrt(
                    np.mean((outputs - model_outputs)**2))
                original_deviation = np.sqrt(
                    np.mean((self.chargingup_outputs[batch_indices] -
                             model_outputs)**2))
                model_deviation_chargeup.append(
                    model_charging_signal_deviations)
                print(
                    f'\tRMSE deviation of measured device output against model output: {model_charging_signal_deviations:.2f} (nA)'
                )
                print(
                    f'\tRMSE deviation of original device output against model output: {original_deviation:.2f} (nA)'
                )
            if batch == 0:
                plt.plot(self.chargingup_outputs[batch_indices],
                         label='Original output',
                         alpha=0.5)
                plt.plot(outputs, label='Measured output', alpha=0.5)
                if self.model is not None:
                    plt.plot(model_outputs, label='Model output', alpha=0.5)
                plt.title(f'Raw comparison of the first batch of signals.')
                plt.legend()
                plt.savefig(os.path.join(self.results_dir, 'first_batch'))
        print('\n')
        # Initialize sampling loop
        for trial in range(self.repetitions):
            start_trial = time.time()
            for batch, batch_indices in enumerate(
                    self.get_batch_indices(self.nr_samples, self.batch_size)):

                # Generate inputs (without ramping)
                inputs = self.reference_inputs[:, batch_indices]
                # Get outputs (without ramping; raming is done in the get_batch method)
                device_outputs = self.sample_batch(inputs)
                results[trial, batch_indices] = device_outputs
                # if (batch % 1) == 0:
                #     self.plot_waves(inputs.T, outputs, batch)
                if self.model is not None:
                    model_outputs = self.get_model_batch(inputs)
                    model_results[trial, batch_indices] = device_outputs
            end_trial = time.time()
            deviations[trial] = np.sqrt(
                np.mean((results[trial] - self.reference_outputs)**2))
            correlation[trial] = np.corrcoef(results[trial].T,
                                             self.reference_outputs.T)[0, 1]
            print(
                f'* Consistency check {trial+1}/{self.repetitions} took {end_trial - start_trial :.2f} sec. with {batch+1} batches'
            )
            print(
                f"\tCorr: {correlation[trial]:.2f} ; Deviation: {deviations[trial]:.2f}"
            )
            if self.model is not None:
                model_deviations[trial] = np.sqrt(
                    np.mean((results[trial] - model_results[trial])**2))
                model_correlation[trial] = np.corrcoef(
                    results[trial].T, model_results[trial].T)[0, 1]
        self.driver.close_tasks()
        #self.close_processor()

        np.savez(self.results_filename,
                 results=results,
                 deviations=deviations,
                 correlation=correlation)
        print(f'Data saved in { self.results_filename}')
        if self.model is None:
            return results, deviations, correlation, deviation_chargeup
        else:
            return results, deviations, correlation, deviation_chargeup, model_results, model_deviations, correlation, deviation_chargeup

    def get_model_batch(self, input_batch):
        # outputs_device = super(ConsistencyChecker,self).sample_batch(input_batch)
        # Ramp input batch (0.5 sec up and down) and format it to pytorch
        batch_ramped = TorchUtils.format(self.ramp_input(input_batch).T)
        self.model.eval()
        with torch.no_grad():
            outputs_ramped = TorchUtils.to_numpy(
                self.model(batch_ramped))  #.squeeze(0)
        if len(outputs_ramped.shape) > 1:
            return outputs_ramped[self.filter_ramp[:, np.newaxis]][:,
                                                                   np.newaxis]


def consistency_check(main_dir,
                      repetitions=1,
                      sampler_configs_name='sampler_configs.json',
                      reference_batch_name='reference_batch.npz',
                      charging_signal_name='charging_signal.npz',
                      model=None):

    sampler = ConsistencyChecker(main_dir,
                                 repetitions=repetitions,
                                 sampler_configs_name=sampler_configs_name,
                                 reference_batch_name=reference_batch_name,
                                 charging_signal_name=charging_signal_name,
                                 model=model)
    if model is None:
        outputs, deviations, correlation, deviation_chargeup = sampler.get_data(
        )
    else:
        outputs, deviations, correlation, deviation_chargeup, model_outputs, model_deviations, model_correlation, model_deviation_chargeup = sampler.get_data(
        )
    mean_output = np.mean(outputs, axis=0)
    std_output = np.std(outputs, axis=0)
    if model is not None:
        model_mean_output = np.mean(model_outputs, axis=0)
        model_std_output = np.mean(model_outputs, axis=0)
    plt.figure()
    plt.plot(sampler.reference_outputs,
             'r',
             label='reference signal',
             alpha=0.5)
    plt.plot(mean_output, 'k', label='mean over repetitions', alpha=0.5)
    plt.plot(mean_output + std_output, ':k', alpha=0.5)
    plt.plot(mean_output - std_output,
             ':k',
             label='stdev over repetitions',
             alpha=0.5)
    if model is not None:
        plt.plot(model_mean_output, 'c', label='Model output', alpha=0.5)
        # It could be removed with if model.noise is None
        plt.plot(model_mean_output + model_std_output, ':c', alpha=0.5)
        plt.plot(model_mean_output + model_std_output,
                 ":c",
                 label='Stdev of the model')
    plt.title(f'Consistency over {sampler.repetitions} trials with same input')
    plt.legend()
    plt.savefig(os.path.join(sampler.results_dir, 'consistency_check'))

    plt.figure()
    plt.plot(mean_output - sampler.reference_outputs,
             "b",
             label="mean - reference")
    plt.plot(mean_output - sampler.reference_outputs + std_output,
             ":k",
             label="stdev over repetitions")
    plt.plot(mean_output - sampler.reference_outputs - std_output, ":k")
    # if model is not None:
    #     plt.plot(model_mean_output - sampler.reference_outputs, "c", label)
    plt.title("Difference Mean Signal and Reference Signal (nA)")
    plt.legend()
    plt.savefig(os.path.join(sampler.results_dir, 'diff_mean-ref'))

    plt.figure()
    plt.hist(deviations,
             label="Deviations from the reference signal (nA)",
             alpha=0.5)
    if model is not None:
        plt.hist(model_deviations,
                 label="Deviations from the model (nA)",
                 alpha=0.5)
        plt.title("RMSE Deviations from Reference Signal and model (nA)")
        plt.legend()
    else:
        plt.title("RMSE Deviations from Reference Signal (nA)")
    plt.savefig(os.path.join(sampler.results_dir, 'hist_deviations'))

    plt.figure()
    plt.plot(deviation_chargeup, label='Device', alpha=0.5)
    if model is not None:
        plt.plot(model_deviation_chargeup, label='Device vs Model', alpha=0.5)
    plt.title("RMSE deviations (nA) while charging up")
    plt.savefig(
        os.path.join(sampler.results_dir, 'deviations_while_charging_up'))

    plt.show()


if __name__ == '__main__':
    import torch
    from brainspy.processors.processor import Processor
    from brainspy.utils.pytorch import TorchUtils

    configs = {}
    configs["processor_type"] = "simulation"
    # configs["input_indices"] = [2, 3]
    configs["electrode_effects"] = {}
    # configs["electrode_effects"]["amplification"] = [1]
    # configs["electrode_effects"]["output_clipping"] = [-114, 114]
    # configs["electrode_effects"]["noise"] = {}
    # configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
    # configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
    configs["driver"] = {}
    configs["waveform"] = {}
    configs["waveform"]["plateau_length"] = 1
    configs["waveform"]["slope_length"] = 0

    model_data = torch.load(
        'C:/Users/Unai/Documents/programming/smg/tmp/output/conv_model/training_data_2021_07_19_143513/training_data.pt'
    )
    model = Processor(configs, model_data['info'],
                      model_data['model_state_dict'])
    model = TorchUtils.format(model)
    consistency_check(
        'C:/Users/Unai/Documents/programming/smg/tmp/data/training/TEST/Brains_testing_2021_07_16_093337',
        repetitions=10,
        model=model)
