"""
File containing a class for checking a device (or/and a surrogate model) against previously obtained reference measurements.
"""
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs
from bspysmg.data.sampling import Sampler
from brainspy.utils.io import create_directory_timestamp
from brainspy.utils.pytorch import TorchUtils
from typing import Tuple


class ConsistencyChecker(Sampler):
    def __init__(self,
                 main_dir: str,
                 repetitions: int = 1,
                 sampler_configs_name: str = 'sampler_configs.json',
                 reference_batch_name: str = 'reference_batch.npz',
                 charging_signal_name: str = 'charging_signal.npz',
                 model: torch.nn.Module = None,
                 show_plots: bool = True) -> None:
        """
        Initializes dataset and directory to save results for consistency checking
        experiment of a model. This function uses sampler config files and already
        existing device input-output dataset and original measurements.

        Parameters
        ----------
        main_dir : str
            Path to main directory which contains the configuration files.
        repetitions : int [Optional]
            Number of times the experiments should be repeated.
        sampler_configs_name : str [Optional]
            Name of the file which contains sampling configuration and is used to initialize the
            parent class Sampler. It has the following keys:
            * save_directory: str
                Directory where the all the sampling data will be stored.
            * data_name: str
                Inside the path specified on the variable save_directory, a folder will be created,
                with the format: <data_name>+<current_timestamp>. This variable specified the
                prefix of that folder before the timestamp.
            * driver: dict
                Dictionary containing the driver configurations. For more information check the
                documentation about this configuration file, check the documentation of
                brainspy.processors.hardware.drivers.ni.setup.NationalInstrumentsSetup
            * input_data : dict
                Dictionary containing the information necessary to create the input sampling data.
                - input_distribution: str
                    It determines the wave shape of the input. Two main options availeble 'sawtooth'
                    and 'sine'. The first option will create saw-like signals, and the second
                    sine-wave signals. Sawtooth signals have more coverage on the edges of the
                    input range.
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
                - factor : float
                    Given factor by which the input frequencies will be multiplied after square
                    rooting them.
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
                - ramp_time: float
                    Time that will be taken before sending each batch to go from zero to the first
                    point of the batch and to zero from the last point of the batch.
                - batch_time:
                    Time that the sampling of each batch will take.
                - number_batches: int
                    Number of batches that will be sampled. A default value of 3880 is reccommended.
        reference_batch_name : str [Optional]
            Name of the file which contains the reference dataset. This is the original device
            measurements. It is a npz file with columns 'inputs' and 'outputs'.
        charging_signal_name : str [Optional]
            Name of the file which contains device inputs and outputs. This is the device behaviour
            at present moment. It is a npz file with columns 'inputs' and 'outputs'.
        model : custom model of type torch.nn.Module [Optional]
            Model whose consistency is to be checked. This is a trained neural network model over
            DNPU measurements and sampled input data.
        """
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

    def get_data(self, charge_device=True) -> Tuple[np.array]:
        """
        The main function that implements consistency checking routine. It uses
        the reference data and device's outputs to check if the outputs of device
        are consistent with device over several runs. Optionally it can also check
        consistency of a trained neural network over device measurements.

        Returns
        -------
        tuple
            tuple with following data:
                - results: np.array
                    outputs generated by the device.
                - deviations: np.array
                    RMSE between device outputs and refernce data.
                - correlation: np.array
                    Value of correlation coefficient between device outputs and
                    refence data.
                - deviation_chargeup: np.array
                    RMSE deviation between device output and original device output.
                
                if model is not None:
                    - model_results: np.array
                        outputs generated by the model.
                    - model_deviations: np.array
                        RMSE between model outputs and refernce data.
                    - model_correlation: np.array
                        Value of correlation coefficient between model outputs and
                        refence data.
                    - model_deviation_chargeup: np.array
                        RMSE deviation between model output and original device output.
        """
        results = np.zeros((self.repetitions, ) + self.reference_outputs.shape)
        deviations = np.zeros(self.repetitions)
        correlation = np.zeros(self.repetitions)
        deviation_chargeup = []
        if self.model is not None:
            model_results = results.copy()
            model_deviations = deviations.copy()
            model_correlation = correlation.copy()
            model_deviation_chargeup = []
            if charge_device:
                self.charge_device(deviation_chargeup,
                                   model_deviation_chargeup)
        else:
            if charge_device:
                self.charge_device(deviation_chargeup)
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
                    model_outputs = self.sample_model_batch(inputs)
                    model_results[trial, batch_indices] = model_outputs
            end_trial = time.time()
            deviations[trial] = np.sqrt(
                np.mean((results[trial] - self.reference_outputs)**2))
            correlation[trial] = np.corrcoef(results[trial].T,
                                             self.reference_outputs.T)[0, 1]
            print(
                f'* Consistency check {trial+1}/{self.repetitions} took {end_trial - start_trial :.2f} sec. with {batch+1} batches'
            )
            print(
                f"\tCorr: {correlation[trial]:.2f} ; RMSE Deviation: {deviations[trial]:.2f}"
            )
            if self.model is not None:
                model_deviations[trial] = np.sqrt(
                    np.mean((results[trial] - model_results[trial])**2))
                model_correlation[trial] = np.corrcoef(
                    results[trial].T, model_results[trial].T)[0, 1]
                print(
                    f"\tCorr (Model): {model_correlation[trial]:.2f} ; RMSE Deviation (Model): {model_deviations[trial]:.2f}"
                )
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
            return results, deviations, correlation, deviation_chargeup, model_results, model_deviations, model_correlation, model_deviation_chargeup

    def charge_device(self, deviation_chargeup, model_deviation_chargeup=None, show_plots=True):
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
                model_outputs = self.sample_model_batch(inputs)
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
                plt.title(f'Charging signal (Batch 0)')
                plt.legend()
                plt.savefig(os.path.join(self.results_dir, 'first_batch'))
        print('Finished charging up device. \n')

    # def get_batch(self, input_batch):
    #      super(ConsistencyChecker, self).get_batch(input_batch)
    #     # # Ramp input batch (0.5 sec up and down)
    #     # batch_ramped = self.ramp_input_batch(input_batch)
    #     # # Readout output signal
    #     # outputs_ramped = self.driver.forward_numpy(batch_ramped.T)
    #     return outputs_device[self.filter_ramp]

    def sample_model_batch(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        Ramp the input batch (0.5 sec up and down) and format it to Tensor. Ramping is
        required to avoid abrupt changes in the voltage. The input is masked from 0v to
        first value and then final value back to 0v.

        Returns
        -------
        torch.Tensor
            Input tensor ramped and allocated to tensor.
        """
        #outputs_device = super(ConsistencyChecker,self).get_batch(input_batch)
        # outputs_device = super(ConsistencyChecker,self).sample_batch(input_batch)
        # Ramp input batch (0.5 sec up and down) and format it to pytorch
        #batch_ramped = TorchUtils.format(self.ramp_input(input_batch).T)
        self.model.eval()
        with torch.no_grad():
            outputs = TorchUtils.to_numpy(
                self.model(TorchUtils.format(input_batch).T))  #.squeez0e(0)
        # if len(outputs_ramped.shape) > 1:
        #     return outputs_ramped[self.filter_ramp[:, np.newaxis]][:,
        #                                                            np.newaxis]
        return outputs


def consistency_check(main_dir: str,
                      repetitions: int = 1,
                      sampler_configs_name: str = 'sampler_configs.json',
                      reference_batch_name: str = 'reference_batch.npz',
                      charging_signal_name: str = 'charging_signal.npz',
                      charge_device: bool = True,
                      model: torch.nn.Module = None,
                      show_plots = True) -> None:
    """
    This is the driver function used for consistency checking. Consistency checking involves
    checking how DNPU device is behaving at present moment against how it was before
    measurement. This check can also be performed against trained neural network model over
    DNPU measurements. This function initializes a ConsistencyChecker object and performs the 
    consistency check using the get_data function. It also plots and saves various graphs of
    calculated metrics.

    Parameters
    ----------
    main_dir : str
        Path to main directory which contains the configuration files.
    repetitions : int [Optional]
        Number of times the experiments should be repeated.
    sampler_configs_name : str [Optional]
        Name of the file which contains sampling configuration with keys mentioned in
        constructor function of ConsistencyChecker class.
    reference_batch_name : str [Optional]
        Name of the file which contains the reference dataset. This is the original device
        measurements. It is a npz file with columns 'inputs' and 'outputs'.
    charging_signal_name : str [Optional]
        Name of the file which contains device inputs and outputs. This is the device behaviour
        at present moment. It is a npz file with columns 'inputs' and 'outputs'.
    charge_device: boolean [Optional]
        Whether the consistency check should charge up the device with the charging signal or not.
    model : custom model of type torch.nn.Module [Optional]
        Model whose consistency is to be checked. This is a trained neural network model over
        DNPU measurements and sampled input data.
    """
    sampler = ConsistencyChecker(main_dir,
                                 repetitions=repetitions,
                                 sampler_configs_name=sampler_configs_name,
                                 reference_batch_name=reference_batch_name,
                                 charging_signal_name=charging_signal_name,
                                 model=model)
    if model is None:
        outputs, deviations, correlation, deviation_chargeup = sampler.get_data(
            charge_device)
    else:
        outputs, deviations, correlation, deviation_chargeup, model_outputs, model_deviations, model_correlation, model_deviation_chargeup = sampler.get_data(
            charge_device)
    mean_output = np.mean(outputs, axis=0)
    std_output = np.std(outputs, axis=0)
    plt.figure()
    plt.plot(sampler.reference_outputs,
             'r',
             label='Reference signal',
             alpha=0.5)
    plt.plot(mean_output, 'k', label='Device output (mean)', alpha=0.5)
    plt.plot(mean_output + std_output, ':k', alpha=0.5)
    plt.plot(mean_output - std_output,
             ':k',
             label='Device output (std)',
             alpha=0.5)
    if model is not None:
        model_mean_output = np.mean(model_outputs, axis=0)
        model_std_output = np.std(model_outputs, axis=0)
        plt.plot(model_mean_output,
                 'c',
                 label='Model output (mean)',
                 alpha=0.5)
        # It could be removed with if model.noise is None
        plt.plot(model_mean_output + model_std_output, ':c', alpha=0.5)
        plt.plot(model_mean_output - model_std_output,
                 ":c",
                 label='Model output (std)')
    plt.title(f'Reference signal over {sampler.repetitions} trials')
    plt.legend()
    plt.savefig(os.path.join(sampler.results_dir, 'consistency_check'))

    plt.figure()
    plt.hist(np.sqrt((mean_output - sampler.reference_outputs)**2),
             bins=100,
             alpha=0.5,
             label='Device')
    if model is not None:
        plt.hist(np.sqrt((model_mean_output - sampler.reference_outputs)**2),
                 bins=100,
                 alpha=0.5,
                 label="Model")
    plt.title(
        "Reference Signal\nHistogram of RMSE deviations (nA) from mean over " +
        str(repetitions) + " trials")
    plt.legend()
    # plt.figure()
    # plt.plot(mean_output - sampler.reference_outputs,
    #          "b",
    #          label="Error over mean (Device)", alpha=0.5)
    # plt.plot(mean_output - sampler.reference_outputs + std_output,
    #          ":b",
    #          label="Error over std (Device) ", alpha=0.5)
    # plt.plot(mean_output - sampler.reference_outputs - std_output, ":b"
    # , alpha=0.5)
    # if model is not None:
    #     plt.plot(model_mean_output - sampler.reference_outputs, "c",
    #     label="Error over mean (Model)", alpha=0.5)
    #     plt.plot(model_mean_output - sampler.reference_outputs + model_std_output,
    #     ":c",
    #     label="Error over std (Model) ", alpha=0.5)
    #     plt.plot(model_mean_output - sampler.reference_outputs + model_std_output,
    #     ":c", alpha=0.5)

    # plt.title("Reference Signal (Error)")
    # plt.legend()
    # plt.savefig(os.path.join(sampler.results_dir, 'diff_mean-ref'))

    if charge_device:
        plt.figure()
        plt.plot(deviation_chargeup, label='Device', alpha=0.5)
        if model is not None:
            plt.plot(model_deviation_chargeup,
                     label='Device vs Model',
                     alpha=0.5)
        plt.title("Charging up signal")
        plt.xlabel("Batch number of signal")
        plt.ylabel("RMSE Current (nA)")
        plt.legend()
        plt.savefig(
            os.path.join(sampler.results_dir, 'deviations_while_charging_up'))
    if show_plots:
        plt.show()


# if __name__ == '__main__':
#     import torch
#     from brainspy.processors.processor import Processor
#     from brainspy.utils.pytorch import TorchUtils

#     configs = {}
#     configs["processor_type"] = "simulation"
#     # configs["input_indices"] = [2, 3]
#     configs["electrode_effects"] = {}
#     # configs["electrode_effects"]["amplification"] = [1]
#     # configs["electrode_effects"]["output_clipping"] = [-114, 114]
#     # configs["electrode_effects"]["noise"] = {}
#     # configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
#     # configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
#     configs["driver"] = {}
#     configs["waveform"] = {}
#     configs["waveform"]["plateau_length"] = 1
#     configs["waveform"]["slope_length"] = 0

#     # model_data = torch.load(
#     #     'C:/Users/Unai/Documents/programming/smg/tmp/output/conv_model/training_data_2021_07_19_143513/training_data.pt'
#     # )

#     configs = {}
#     configs['processor_type'] = 'simulation'
#     configs["waveform"] = {}
#     configs["waveform"]["plateau_length"] = 1 #10
#     configs["waveform"]["slope_length"] = 0 #30

#     model_data = {}
#     model_data["info"] = {}
#     model_data["info"]["model_structure"] = {
#         "hidden_sizes": [90, 90, 90],
#         "D_in": 7,
#         "D_out": 1,
#         "activation": "relu",
#     }
#     model_data["info"]['electrode_info'] = {}
#     model_data["info"]['electrode_info']['electrode_no'] = 8
#     model_data["info"]['electrode_info']['activation_electrodes'] = {}
#     model_data["info"]['electrode_info']['activation_electrodes']['electrode_no'] = 7
#     model_data["info"]['electrode_info']['activation_electrodes'][
#             'voltage_ranges'] = np.array([[-0.55, 0.325], [-0.95, 0.55],
#                                           [-1., 0.6], [-1., 0.6], [-1., 0.6],
#                                           [-0.95, 0.55], [-0.55, 0.325]])
#     model_data["info"]['electrode_info']['output_electrodes'] = {}
#     model_data["info"]['electrode_info']['output_electrodes']['electrode_no'] = 1
#     model_data["info"]['electrode_info']['output_electrodes']['amplification'] = [28.5]
#     model_data["info"]['electrode_info']['output_electrodes']['clipping_value'] = None

#     model = Processor(configs, model_data['info'])#,
#                       #model_data['model_state_dict'])
#     model = TorchUtils.format(model)
#     consistency_check(
#         'tests/data/',
#         repetitions=1,
#         model=model)
