import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from brainspy.utils.io import load_configs
from brainspy.utils.pytorch import TorchUtils


def data_loader(data_directory):
    config_path = os.path.join(data_directory, "sampler_configs.json")
    configs = load_configs(config_path)
    data_path = os.path.join(data_directory, "IO.dat")
    print("\nLoading file: " + data_path)
    print("This may take some time. Please wait.\n")
    data = np.loadtxt(data_path)
    inputs = data[:, :configs["input_data"]["input_electrodes"]]
    outputs = data[:, -configs["input_data"]["output_electrodes"]:]
    return inputs, outputs, configs


def post_process(data_directory, clipping_value="default", **kwargs):

    """Postprocess data, cleans clipping, and merges data sets if needed. The data arrays are merged
    into a single array and cropped given the clipping_values. The function also plots and saves the histogram of the data
    Arguments:
        - a string with path to the directory with the data: it is assumed there is a
        sampler_configs.json and a IO.dat file.
        - clipping_value : The the setups have a limit in the range they can read. They typically clip at approximately +-4 V.
                Note that in order to calculate the clipping_range, it needs to be multiplied by the amplification value of the setup. (e.g., in the Brains setup the amplification is 28.5,
                is the clipping_value is +-4 (V), therefore, the clipping value should be +-4 * 28.5, which is [-110,110] (nA) ).
                This variable represents a lower and upper clipping_value to crop data. It can be either None, 'default' or [float,float].
                The 'default' str input will automatically take the clipping value by multiplying the amplification of the data by -4 and 4. The None input will not apply any clipping.
                [float,float] will apply a clipping within the specified values.
    Optional kwargs:
        - list_data: A list of strings indicating directories with training_NN_data.npz containing 'data'.

    NOTE:
        - The data is saved in path_to_data to a .npz file with keyes: inputs, outputs and info,
        which has a dictionary with the configs of the sampling procedure.
        - The inputs are on ALL electrodes in Volts and the output in nA.
        - Data does not undergo any transformation, this is left to the user.
        - Data structure of output and input are arrays of NxD, where N is the number of samples and
        D is the dimension.
    """
    # Load full data
    if not list(kwargs.keys()):
        inputs, outputs, configs = data_loader(data_directory)
    else:  # Merge data if list_data is in kwargs
        if "list_data" in kwargs.keys():
            inputs, outputs, configs = data_merger(data_directory, kwargs["list_data"])
        else:
            assert (
                False
            ), f"{list(kwargs.keys())} not recognized! kwargs must be list_data"
    batch_length = (
        configs["input_data"]["batch_time"]
        * configs["driver"]["sampling_frequency"]
    )
    nr_raw_samples = len(outputs)
    print("Number of raw samples: ", nr_raw_samples)
    assert (
        nr_raw_samples == configs["input_data"]["number_batches"] * batch_length
    ), f"Data size mismatch!"
    output_scales = [np.min(outputs), np.max(outputs)]
    print(f"Output scales: [Min., Max.] = {output_scales}")
    # input_scales = list(zip(np.min(inputs, axis=0), np.max(inputs, axis=0)))
    print(f"Lower bound input scales: {np.min(inputs,axis=0)}")
    print(f"Upper bound input scales: {np.max(inputs,axis=0)}\n")
    # Get charging signals
    charging_batches = int(
        60 * 30 / configs["input_data"]["batch_time"]
    )  # ca. 30 min charging signal
    save_npz(
        data_directory,
        "charging_signal",
        inputs[-charging_batches * batch_length :],
        outputs[-charging_batches * batch_length :],
        configs,
    )
    # Get reference batches
    refs_batches = int(
        600 / configs["input_data"]["batch_time"]
    )  # ca. 600s reference signal
    save_npz(
        data_directory,
        "reference_batch",
        inputs[-refs_batches * batch_length :],
        outputs[-refs_batches * batch_length :],
        configs,
    )
    # Plot samples histogram and save
    output_hist(outputs[::3], data_directory, bins=1000)

    # Clean data
    configs["electrode_info"] = get_electrode_info(configs, clipping_value)
    if configs["electrode_info"]["output_electrodes"]["clipping_value"] is not None:
        inputs, outputs = clip_data(
            inputs,
            outputs,
            configs["electrode_info"]["output_electrodes"]["clipping_value"],
        )
        print("% of points cropped: ", (1 - len(outputs) / nr_raw_samples) * 100)
        print("\n")
    # save data
    save_npz(data_directory, "postprocessed_data", inputs, outputs, configs)

    return inputs, outputs, configs


def save_npz(data_directory, file_name, inputs, outputs, configs):
    save_to = os.path.join(data_directory, file_name)
    print(f"Data saved to: {save_to}.npz")
    np.savez(save_to, inputs=inputs, outputs=outputs, info=configs)


def get_electrode_info(configs, clipping_value):
    electrode_info = {}
    electrode_info["electrode_no"] = (
        configs["input_data"]["input_electrodes"]
        + configs["input_data"]["output_electrodes"]
    )
    electrode_info["activation_electrodes"] = {}
    electrode_info["activation_electrodes"]["electrode_no"] = configs["input_data"][
        "input_electrodes"
    ]
    electrode_info["activation_electrodes"]["voltage_ranges"] = get_voltage_ranges(
        configs
    )
    electrode_info["output_electrodes"] = {}
    electrode_info["output_electrodes"]["electrode_no"] = configs["input_data"][
        "output_electrodes"
    ]
    electrode_info["output_electrodes"]["amplification"] = configs[
        "driver"
    ]["amplification"]
    if clipping_value == "default":
        electrode_info["output_electrodes"]["clipping_value"] = (
            electrode_info["output_electrodes"]["amplification"] * np.array([-4, 4])
        ).tolist()
    else:
        electrode_info["output_electrodes"]["clipping_value"] = clipping_value

    print_electrode_info(electrode_info)
    return electrode_info


def get_voltage_ranges(configs):
    offset = np.array(configs["input_data"]["offset"])
    amplitude = np.array(configs["input_data"]["amplitude"])
    min_voltage = (offset - amplitude)[:, np.newaxis]
    max_voltage = (offset + amplitude)[:, np.newaxis]
    return np.concatenate((min_voltage, max_voltage), axis=1)


def print_electrode_info(configs):
    print(
        f"\nThe following data is inferred from the input data. Please check if it is correct. "
    )
    print(
        f"Data is gathered from a device with {configs['electrode_no']} electrodes, from which: "
    )
    print(
        f"There are {configs['activation_electrodes']['electrode_no']} activation electrodes: "
    )
    print(
        "\t * Lower bound of voltage ranges: "
        + str(configs["activation_electrodes"]["voltage_ranges"][:, 0])
    )
    print(
        "\t * Upper bound of voltage ranges: "
        + str(configs["activation_electrodes"]["voltage_ranges"][:, 1])
    )
    print(
        f"There are {configs['output_electrodes']['electrode_no']} output electrodes: "
    )
    print("\t * Clipping value: " + str(configs["output_electrodes"]["clipping_value"]))
    print(
        "\t * Amplification correction value: "
        + str(configs["output_electrodes"]["amplification"])
    )


def output_hist(outputs, data_directory, bins=100, show=False):
    plt.figure()
    plt.title("Output Histogram")
    plt.hist(outputs, bins=bins)
    plt.ylabel("Counts")
    plt.xlabel("outputs (nA)")
    if show:
        plt.show()
    plt.savefig(data_directory + "/output_distribution")
    plt.close()


def clip_data(inputs, outputs, clipping_value):

    print(f"\nClipping data outside range {clipping_value[0]} and {clipping_value[1]}")
    mean_output = np.mean(outputs, axis=1)
    # Get cropping mask
    if type(clipping_value) is list:
        cropping_mask = (mean_output < clipping_value[1]) * (
            mean_output > clipping_value[0]
        )
    elif type(clipping_value) is float:
        cropping_mask = np.abs(mean_output) < clipping_value
    else:
        TypeError(
            f"Clipping value not recognized! Must be list with lower and upper bound or float, was {type(clipping_value)}"
        )

    outputs = outputs[cropping_mask]
    inputs = inputs[cropping_mask, :]
    return inputs, outputs


############################################################################
# TODO:


def data_merger(list_dirs):
    NotImplementedError(
        "Merging of data from a list of data directories not implemented!"
    )
    # raw_data = {}
    # out_list = []
    # inp_list = []
    # meta_list = []
    # datapth_list = []
    # for dir_file in list_dirs:
    #     _databuff = data_loader(main_dir + dir_file)
    #     out_list.append(_databuff['outputs'])
    #     meta_list.append(_databuff['meta'])
    #     datapth_list.append(_databuff['data_path'])
    #     inp_list.append(_databuff['inputs'])
    # # Generate numpy arrays out of the lists
    # raw_data['outputs'] = np.concatenate(tuple(out_list))
    # raw_data['inputs'] = np.concatenate(tuple(inp_list))
    # raw_data['meta'] = meta_list
    # raw_data['data_path'] = datapth_list


if __name__ == "__main__":
    import matplotlib

    # matplotlib.use('TkAgg')
    main_dir = "/home/unai/Documents/3-Programming/bspy/smg/tmp/data/training/TEST/17-02-2021/"

    # dirs = list(
    #     [
    #         name
    #         for name in os.listdir(main_dir)
    #         if os.path.isdir(os.path.join(main_dir, name)) and not name.startswith(".")
    #     ]
    # )

    # assert len(dirs) > 0
    # for i in range(len(dirs)):
    inputs, outputs, info = post_process(main_dir)
    # output_hist(outputs, os.path.join(main_dir, dirs[i]), bins=1000, show=True)
