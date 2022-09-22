"""
File containing functions for postprocessing raw data gathered from the sampler and information for the model's info dictionary.
"""
import os

import numpy as np

from brainspy.utils.io import load_configs
from bspysmg.utils.plots import output_hist
from typing import Tuple
from datetime import timedelta


def get_sampling_data(filename: str, activation_electrode_no: int,
                      readout_electrode_no: int) -> Tuple[np.array, np.array]:
    """
    Reads the sampling data from a text file (IO.dat) and returs the values loaded in numpy arrays.

    Parameters
    ----------
    filename : str
        Path to the file containing comma separated values read during the data gathering
        process. Typically, named IO.dat.
    activation_electrode_no : int
        Number of activation electrodes used for the device during the data gathering process.
    readout_electrode_no : int
        Number of current readout/output electrodes used for the device during the data gathering
        process.

    Returns
    -------
    inputs : np.array
        Array containing all the inputs that were sent to the device during sampling.
    outputs : np.array
        Array containing all the outputs of the device obtained during sampling, which correspond
        to the inputs to the device.
    """
    print("\nLoading file: " + filename)
    print("This may take some time. Please wait.\n")
    assert type(activation_electrode_no
                ) is int, "Activation electrode number expected to be int"
    assert type(readout_electrode_no
                ) is int, "Readout electrode number expected to be int"
    data = np.loadtxt(filename)
    assert data.shape[1] == (
        activation_electrode_no + readout_electrode_no
    ), "Data from the file has a different electrode configuration. Check the activation electrode no and the readout electrode no"
    inputs = data[:, :activation_electrode_no]
    outputs = data[:, -readout_electrode_no:]
    return inputs, outputs


def post_process(data_dir: str,
                 clipping_value="default",
                 charging_signal_batch_no: int = 40,
                 reference_signal_batch_no: int = 15,
                 filename: str = "postprocessed_data",
                 **kwargs) -> Tuple[np.array, np.array, dict]:
    """
    Postprocesses the data, cleans any clipping (optional), and merges data sets if needed. The data
    arrays are merged into a single array and cropped given the clipping_values. The function also
    plots and saves the histogram of the data.

    Parameters
    ----------
    data_dir: str
        A string with path to the directory with the data: it is assumed at least two
        files exist, named sampler_configs.json and a IO.dat respectively.
    clipping_value : [float,float]
        Will apply a clipping to the input and output sampling data within the
        specified values. The the setups have a limit in the range they can read.
        They typically clip at approximately +-4 V. Note that in order to
        calculate the clipping_range, it needs to be multiplied by the
        amplification value of the setup. (e.g., in the Brains setup the
        amplification is 28.5, is the clipping_value is +-4 (V), therefore, the
        clipping value should be +-4 * 28.5, which is [-110,110] (nA) ).
        This variable represents a lower and upper clipping_value to crop data.
        It can be either None, 'default' or [float,float]. The 'default' str
        input will automatically take the clipping value by multiplying the
        amplification of the data by -4 and 4. The None input will not apply any
        clipping. 
        
        N O T E: When the clipping value is set to None, the model will accurately
        represent the hardware setup (feedback resistance of the operational
        amplifier). When clipping value set to the values that
        are clipping, the model will extrapolate the results outside of the clipping
        range caused by the hardaware setup.
    charging_signal_batch_no: [int]
        Number of batches that will be used for extracting the charging signal.
    reference_signal_batch_no: [int]
        Number of batches that will be used for extracting the reference signal.
    filename: [str]
        The name of the file that will be produced after postprocessing. By default: postprocessed_data.npz
    kwargs: Optional kwargs are as follows:
        1. list_data: A list of strings indicating directories with postprocessed_data.npz
        containing input and output data relationships from the device, as well
        as the configuration with which the data was acquired.

    Examples
    --------

    >>> inputs, outputs, configs = post_process('tmp/data/training/TEST/17-02-2021/')

    Notes
    -----
    The postprocessed data is a .npz file called postprocessed_data.npz
    with keys: inputs, outputs and info (dict)

    1. inputs: np.array
    The input(s) is(are) gathered for all activation electrodes. The units is in Volts.

    2. outputs: The output(s) is(are) gathered from all the readout electrodes. The units are in nA.
    The output data is raw. Additional amplification correction might be needed, this is
    left for the user to decide.

    3. info: dict
    Data structure of output and input are arrays of NxD, where N is the number of samples
    and D is the dimension.

    The configs dictionary contains a copy of the configurations used for sampling the data.
    In addition, the configs dictionary has a key named electrode_info, which is created
    during the postprocessing step. The electrode_info key contains the following keys:
    3.1 electrode_no: int
    Total number of electrodes in the device

    3.2 activation_electrodes: dict

    3.2.1 electrode_no: int
    Number of activation electrodes used for gathering the data

    3.2.2 voltage_ranges: list
    Voltage ranges used for gathering the data. It contains the ranges per
    electrode, where the shape is (electrode_no,2). Being 2 the minimum and
    maximum of the ranges, respectively.

    3.3 output_electrodes: dict
    
    3.3.1 electrode_no : int
    Number of output electrodes used for gathering the data

    3.3.2 clipping_value: list[float,float]
    Value used to apply a clipping to the sampling data within the specified
    values.

    3.3.3 amplification: float
    Amplification correction factor used in the device to correct the
    amplification applied to the output current in order to convert it into
    voltage before its readout.

    """
    assert type(charging_signal_batch_no
                ) is int, "charging_signal_batch_no should be an integer"
    assert type(reference_signal_batch_no
                ) is int, "reference_signal_batch_no should be an integer"
    assert data_dir is not None
    configs = load_configs(os.path.join(data_dir, "sampler_configs.json"))
    activation_electrode_no = configs["input_data"]["activation_electrode_no"]
    readout_electrode_no = configs["input_data"]["readout_electrode_no"]

    # If the data comes from multiple sources. Merge them first.
    # if "list_data" in kwargs.keys():
    #     inputs, outputs, configs = data_merger(
    #         data_dir,
    #         kwargs["list_data"],
    #         activation_electrode_no=activation_electrode_no,
    #         readout_electrode_no=readout_electrode_no)
    # elif len(kwargs.keys()) > 0:
    #     assert (
    #         False
    #     ), f"{list(kwargs.keys())} not recognized! kwargs must be list_data"

    inputs, outputs = get_sampling_data(
        os.path.join(data_dir, "IO.dat"),
        activation_electrode_no=activation_electrode_no,
        readout_electrode_no=readout_electrode_no)

    batch_length = int(
        configs["input_data"]["batch_time"] *
        configs["driver"]["instruments_setup"]["activation_sampling_frequency"]
    )
    nr_raw_samples = len(outputs)
    print("Number of raw samples: ", nr_raw_samples)
    assert (nr_raw_samples == configs["input_data"]["number_batches"] *
            batch_length), "Data size mismatch!"
    output_scales = [np.min(outputs), np.max(outputs)]
    print(f"Output scales: [Min., Max.] = {output_scales}")
    # input_scales = list(zip(np.min(inputs, axis=0), np.max(inputs, axis=0)))
    print(f"Lower bound input scales: {np.min(inputs,axis=0)}")
    print(f"Upper bound input scales: {np.max(inputs,axis=0)}\n")
    # Get charging signals
    # charging_batches = int(
    #     60 * 30 /
    #     configs["input_data"]["batch_time"])  # ca. 30 min charging signal
    print("Charging signal contains " + str(charging_signal_batch_no) +
          " batches. Total time: " + str(
              timedelta(seconds=int(charging_signal_batch_no *
                                    configs['input_data']['batch_time']))))
    save_npz(
        data_dir,
        "charging_signal",
        inputs[-charging_signal_batch_no * batch_length:],
        outputs[-charging_signal_batch_no * batch_length:],
        configs,
    )
    # # Get reference batches
    # refs_batches = int(
    #     600 / configs["input_data"]["batch_time"])  # ca. 600s reference signal
    print("\nReference signal contains " + str(reference_signal_batch_no) +
          " batches. Total time: " + str(
              timedelta(seconds=int(reference_signal_batch_no *
                                    configs['input_data']['batch_time']))))
    save_npz(
        data_dir,
        "reference_batch",
        inputs[-reference_signal_batch_no * batch_length:],
        outputs[-reference_signal_batch_no * batch_length:],
        configs,
    )
    # Plot samples histogram and save
    output_hist(outputs[::3], data_dir, bins=100)

    # Clean data
    configs["electrode_info"] = get_electrode_info(configs, clipping_value)
    if configs["electrode_info"]["output_electrodes"][
            "clipping_value"] is not None:
        inputs, outputs = clip_data(
            inputs,
            outputs,
            configs["electrode_info"]["output_electrodes"]["clipping_value"],
        )
        print("% of points cropped: ",
              (1 - len(outputs) / nr_raw_samples) * 100)
        print("\n")
    # save data
    save_npz(data_dir, filename, inputs, outputs, configs)

    return inputs, outputs, configs


def save_npz(data_dir: str, file_name: str, inputs: np.array,
             outputs: np.array, configs: dict) -> None:
    """
    Stores the input, outputs and sampling configurations in an .npz file.
    The saved file needs to be opened with the option pickle=True, since it
    contains a dictionary.

    Parameters
    ----------
    data_dir : str
        Folder where the data is going to be stored.
    file_name : [type]
        The name of the data that wants to be stored.
    inputs : np.array
        Array containing all the inputs that were sent to the device during sampling.
    outputs : np.array
        Array containing all the outputs of the device obtained during sampling, which correspond
        to the inputs to the device.
    configs : dict
        Sampling configurations with the following keys:

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

        4.5 phase : float
        Horizontal shift of the input signals. It is recommended to have random numbers
        which are different for the training, validation and test datasets. These
        numbers will be square rooted and multiplied by a given factor.

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

        4.10 batch_time:
        Time that the sampling of each batch will take.

        4.11 number_batches: int
        Number of batches that will be sampled. A default value of 3880 is reccommended.
    """
    save_to = os.path.join(data_dir, file_name)
    print(f"Data saved to: {save_to}.npz")
    np.savez(save_to, inputs=inputs, outputs=outputs, sampling_configs=configs)


def get_electrode_info(configs: dict, clipping_value) -> dict:
    """
    Retrieve electrode information from the data sampling configurations.

    Parameters
    ----------
    configs : dict
        Sampling configurations with the following keys:
        1. driver: dict
        Dictionary containing the driver configurations. For more information check the
        documentation about this configuration file, check the documentation of
        brainspy.processors.hardware.drivers.ni.setup.NationalInstrumentsSetup
        
        2. input_data : dict
        Dictionary containing the information necessary to create the input sampling data.
        2.1 activation_electrode_no: int
        Number of activation electrodes in the device that wants to be sampled.

        2.2 readout_electrode_no : int
        Number of readout electrodes in the device that wants to be sampled.

        2.3 amplitude : [list[float]]
        Amplitude of the generated input wave signal. It is calculated according to the
        minimum and maximum ranges of each electrode. Where the amplitude value should
        correspond with (max_range_value - min_range_value) / 2. If no amplitude is
        given it will be automatically calculated from the driver configurations for
        activation electrode ranges. If it wants to be manually set, the offset
        variable should also be included in the dictionary.

        2.4 offset: [list[float]]
        Vertical offset of the generated input wave signal. It is calculated according
        to the minimum and maximum ranges of each electrode. Where the offset value
        should correspond with (max_range_value + min_range_value) / 2. If no offset
        is given it will be automatically calculated from the driver configurations for
        activation electrode ranges. If it wants to be manually set, the offset
        variable should also be included in the dictionary.
    clipping_value : str or list
        The value that will be used to clip the sampling data within a specific range. if
        default is passed, a default clipping value will be used. 

    Returns
    -------
    electrode_info : dict
        Configuration dictionary containing all the keys related to the electrode information:
        1. electrode_no: int
        Total number of electrodes in the device

        2. activation_electrodes: dict
        2.1 electrode_no: int
        Number of activation electrodes used for gathering the data

        2.2 voltage_ranges: list
        Voltage ranges used for gathering the data. It contains the ranges per
        electrode, where the shape is (electrode_no,2). Being 2 the minimum and maximum
        of the ranges, respectively.

        3. output_electrodes: dict
        3.1 electrode_no : int
        Number of output electrodes used for gathering the data

        3.2 clipping_value: list[float,float]
        Value used to apply a clipping to the sampling data within the specified values.

        3.3 amplification: float
        Amplification correction factor used in the device to correct the amplification
        applied to the output current in order to convert it into voltage before its
        readout.
    """
    electrode_info = {}
    electrode_info["electrode_no"] = (
        configs["input_data"]["activation_electrode_no"] +
        configs["input_data"]["readout_electrode_no"])
    electrode_info["activation_electrodes"] = {}
    electrode_info["activation_electrodes"]["electrode_no"] = configs[
        "input_data"]["activation_electrode_no"]
    electrode_info["activation_electrodes"][
        "voltage_ranges"] = get_voltage_ranges(
            configs["input_data"]["offset"],
            configs["input_data"]["amplitude"])
    electrode_info["output_electrodes"] = {}
    electrode_info["output_electrodes"]["electrode_no"] = configs[
        "input_data"]["readout_electrode_no"]
    electrode_info["output_electrodes"]["amplification"] = configs["driver"][
        "amplification"]
    if clipping_value == "default":
        electrode_info["output_electrodes"]["clipping_value"] = (
            electrode_info["output_electrodes"]["amplification"] *
            np.array([-4, 4])).tolist()
    else:
        electrode_info["output_electrodes"]["clipping_value"] = clipping_value

    print_electrode_info(electrode_info)
    return electrode_info


def get_voltage_ranges(offset: list, amplitude: list) -> np.array:
    """
    Calculate the voltage ranges of the device out of the information about the
    amplitude and the vertical offset that was used to compute the input waves
    during the data gathering process.

    Parameters
    ----------
    offset : list
        A list of all the offset values to vertically displace the input signal
        in such a way that it fits the activation electrode ranges. The list would
        contain one value per activation electrode.
    amplitude : list
        A list of all the amplitude values to amplify the input signal in such
        a way that it fits the activation electrode ranges.

    Returns
    -------
    np.array
        Array containing the ranges per electrode, where the shape is (electrode_no,2). Being
        2 the minimum and maximum of the ranges, respectively.
    """
    offset = np.array(offset, dtype=np.float32)
    amplitude = np.array(amplitude, dtype=np.float32)
    min_voltage = (offset - amplitude)[:, np.newaxis]
    max_voltage = (offset + amplitude)[:, np.newaxis]
    return np.concatenate((min_voltage, max_voltage), axis=1)


def print_electrode_info(configs: dict) -> None:
    """
    Prints on screen the information about the electrodes that was gathered
    from the configuration file used for gathering the data from the device.

    Parameters
    ----------
    configs : dict
        Configuration dictionary containing all the keys related to the electrode information:
        1. electrode_no: int
        Total number of electrodes in the device

        2. activation_electrodes: dict
        2.1 electrode_no: int
        Number of activation electrodes used for gathering the data

        2.2 voltage_ranges: list
        Voltage ranges used for gathering the data. It contains the ranges per
        electrode, where the shape is (electrode_no,2). Being 2 the minimum and maximum
        of the ranges, respectively.
        3. output_electrodes: dict
        3.1 electrode_no : int
        Number of output electrodes used for gathering the data

        3.2 clipping_value: list[float,float]
        Value used to apply a clipping to the sampling data within the specified values.

        3.3 amplification: float
        Amplification correction factor used in the device to correct the amplification
        applied to the output current in order to convert it into voltage before its
        readout.
    """
    print(
        "\nThe following data is inferred from the input data. Please check if it is correct. "
    )
    print(
        f"Data is gathered from a device with {configs['electrode_no']} electrodes, from which: "
    )
    print(
        f"There are {configs['activation_electrodes']['electrode_no']} activation electrodes: "
    )
    print("\t * Lower bound of voltage ranges: " +
          str(configs["activation_electrodes"]["voltage_ranges"][:, 0]))
    print("\t * Upper bound of voltage ranges: " +
          str(configs["activation_electrodes"]["voltage_ranges"][:, 1]))
    print(
        f"There are {configs['output_electrodes']['electrode_no']} output electrodes: "
    )
    print("\t * Clipping value: " +
          str(configs["output_electrodes"]["clipping_value"]))
    print("\t * Amplification correction value: " +
          str(configs["output_electrodes"]["amplification"]))


def clip_data(inputs: np.array, outputs: np.array,
              clipping_value_range: list) -> Tuple[np.array, np.array]:
    """
    Removes all the outputs and corresponding inputs where the output is outside a given maximum
    and minimum range.

    Parameters
    ----------
    inputs : np.array
        Array containing all the inputs that were sent to the device during sampling.
    outputs : np.array
        Array containing all the outputs of the device obtained during sampling, which correspond
        to the inputs to the device.
    clipping_value_range : list[float,float]
        A list of length two. The first element will be the lower clipping range, and the second
        element will be the higher clipping range.

    Returns
    -------
    inputs : np.array
        Array containing all the inputs that were sent to the device during sampling, except for
        those  values for which its corresponding output is above and below the specified clipping
        range.
    outputs : np.array
        Array containing all the outputs of the device obtained during sampling, except for those
        values for which its corresponding output is above and below the specified clipping range.
    """

    mean_output = np.mean(outputs, axis=1)

    # Get cropping mask
    if type(clipping_value_range) is list:
        cropping_mask = (mean_output < clipping_value_range[1]) * (
            mean_output > clipping_value_range[0])
        print(
            f"\nClipping data outside range {clipping_value_range[0]} and {clipping_value_range[1]}"
        )
        outputs = outputs[cropping_mask]
        inputs = inputs[cropping_mask, :]
        return inputs, outputs
    elif clipping_value_range is None:
        return inputs, outputs
    else:
        raise TypeError(
            f"Clipping value not recognized! Must be list with lower and upper bound or float, was {type(clipping_value_range)}"
        )


# def merge_postprocessed_data(file_names,
#                              output_file_name='merged_postprocessed_data.npz'):
#     """[summary]

#     Parameters
#     ----------
#     file_names : [type]
#         [description]
#     output_file_name : str, optional
#         [description], by default 'merged_postprocessed_data.npz'

#     Example
#     ----------
#     file_names = ['tmp/data/training/Brains_testing_2020_09_04_182557/postprocessed_data.npz',
#      'tmp/data/training/Brains_testing_2020_09_11_093200/postprocessed_data.npz']
#     merge_postprocessed_data(file_names)
#     """
#     ref_data = dict(np.load(file_names[0], allow_pickle='True'))
#     for i in range(1, len(file_names)):
#         data = np.load(file_names[i])
#         for key in list(data):
#             if key != 'info':
#                 ref_data[key] = np.append(ref_data[key], data[key], axis=0)
#     np.savez(output_file_name, **ref_data)

# def data_merger(main_dir, activation_electrode_no=7, readout_electrode_no=1):
#     # EXAMPLE
#     #  main_dir = "tmp/output/model_nips"
#     # The post_process function should have a clipping value which is in an amplified scale.
#     # E.g., for an amplitude of 100 -> 345.5
#     # process_multiple(main_dir)
#     shape = 0
#     dirs = list([
#         name for name in os.listdir(main_dir)
#         if os.path.isdir(os.path.join(main_dir, name))
#         and not name.startswith('.')
#     ])

#     assert len(dirs) > 0
#     for i in range(len(dirs)):
#         shape += np.load(os.path.join(main_dir, dirs[i],
#                                       'postprocessed_data.npz'),
#                          allow_pickle=True)['inputs'].shape[0]

#     input_results = np.zeros([shape, activation_electrode_no])
#     output_results = np.zeros([shape, readout_electrode_no])
#     previous_shape = 0
#     for i in range(len(dirs)):
#         data = np.load(os.path.join(main_dir, dirs[i],
#                                     'postprocessed_data.npz'),
#                        allow_pickle=True)
#         current_shape = previous_shape + data['inputs'].shape[0]
#         input_results[previous_shape:current_shape] = data['inputs']
#         output_results[previous_shape:current_shape] = data['outputs']
#         previous_shape = current_shape
#         info = data['info']

#     info = dict(np.ndenumerate(info))[()]
#     info['input_data']['input_distribution'] = 'mixed'
#     info['input_data']['phase'] = 'mixed'
#     index = np.random.permutation(np.arange(shape))
#     input_results = input_results[index]
#     output_results = output_results[index]

#     limit = int(shape * 0.75)

#     np.savez(os.path.join(main_dir, 'training_data'),
#              inputs=input_results[:limit],
#              outputs=output_results[:limit],
#              info=info)
#     np.savez(os.path.join(main_dir, 'test_data'),
#              inputs=input_results[limit:],
#              outputs=output_results[limit:],
#              info=info)

#if __name__ == "__main__":
# import matplotlib

# matplotlib.use('TkAgg')
# main_dir = "C:/Users/Unai/Documents/github/brainspy-smg/tmp/brains_setup/sampling_data_1KSPS_arsenic_test_2022_03_17_171248"
# inputs, outputs, info = post_process(main_dir,
#                                      clipping_value=[-114, 114],
#                                      filename="postprocessed_data_clipped")
# dirs = list(
#     [
#         name
#         for name in os.listdir(main_dir)
#         if os.path.isdir(os.path.join(main_dir, name)) and not name.startswith(".")
#     ]
# )

# assert len(dirs) > 0
# for i in range(len(dirs)):
# inputs, outputs, info = post_process(main_dir)
# output_hist(outputs, os.path.join(main_dir, dirs[i]), bins=1000, show=True)
