#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:18 2019
@author: hruiz
"""
from bspyalgo.utils.io import load_configs
import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb


def data_loader(data_directory):
    config_path = os.path.join(data_directory, 'sampler_configs.json')
    configs = load_configs(config_path)
    data_path = os.path.join(data_directory, 'IO.dat')
    data = np.loadtxt(data_path)
    inputs = data[:, :configs["input_data"]["input_electrodes"]]
    outputs = data[:, -configs["input_data"]["output_electrodes"]:]
    return inputs, outputs, configs


def post_process(data_directory, clipping_value=[-np.inf, np.inf], **kwargs):
    '''Postprocess data, cleans clipping, and merges data sets if needed. The data arrays are merged
    into a single array and cropped given the clipping_values. The function also plots and saves the histogram of the data
    Arguments:
        - a string with path to the directory with the data: it is assumed there is a
        sampler_configs.json and a IO.dat file.
        - clipping_value (kwarg): A lower and upper clipping_value to crop data; default is [-np.inf,np.inf]
    Optiponal kwargs:
        - list_data: A list of strings indicating directories with training_NN_data.npz containing 'data'.

    NOTE:
        - The data is saved in path_to_data to a .npz file with keyes: inputs, outputs and info,
        which has a dictionary with the configs of the sampling procedure.
        - The inputs are on ALL electrodes in Volts and the output in nA.
        - Data does not undergo any transformation, this is left to the user.
        - Data structure of output and input are arrays of NxD, where N is the number of samples and
        D is the dimension.
    '''
    # Load full data
    if not list(kwargs.keys()):
        inputs, outputs, configs = data_loader(data_directory)
    else:  # Merge data if list_data is in kwargs
        if 'list_data' in kwargs.keys():
            inputs, outputs, configs = data_merger(data_directory, kwargs['list_data'])
        else:
            assert False, f'{list(kwargs.keys())} not recognized! kwargs must be list_data'

    batch_length = configs['input_data']['batch_time'] * configs['processor']['sampling_frequency']
    nr_raw_samples = len(outputs)
    print('Number of raw samples: ', nr_raw_samples)
    assert nr_raw_samples == configs['input_data']['number_batches'] * batch_length, f'Data size mismatch!'
    output_scales = [np.min(outputs), np.max(outputs)]
    print(f'Output scales: [Min., Max.] = {output_scales}')
    input_scales = list(zip(np.min(inputs, axis=0), np.max(inputs, axis=0)))
    print(f'Input scales: {input_scales}')
    # Get charging signals
    charging_batches = int(60 * 30 / configs['input_data']['batch_time'])  # ca. 30 min charging signal
    save_npz(data_directory, 'charging_signal',
             inputs[-charging_batches * batch_length:], outputs[-charging_batches * batch_length:], configs)
    # Get reference batches
    refs_batches = int(600 / configs['input_data']['batch_time'])  # ca. 600s reference signal
    save_npz(data_directory, 'reference_batch',
             inputs[-refs_batches * batch_length:], outputs[-refs_batches * batch_length:], configs)
    # Plot samples histogram and save
    output_hist(outputs[::3], data_directory, bins=500)
    # Clean data
    configs['clipping_value'] = clipping_value
    inputs, outputs = prepare_data(inputs, outputs, clipping_value)
    print('% of points cropped: ', (1 - len(outputs) / nr_raw_samples) * 100)
    # save data
    save_npz(data_directory, 'postprocessed_data', inputs, outputs, configs)

    return inputs, outputs, configs


def save_npz(data_directory, file_name, inputs, outputs, configs):
    save_to = os.path.join(data_directory, file_name)
    print(f'Data saved to \n {save_to}')
    np.savez(save_to, inputs=inputs, outputs=outputs, info=configs)


def output_hist(outputs, data_directory, bins=100):
    plt.figure()
    plt.suptitle('Output Histogram')
    plt.hist(outputs, bins)
    plt.ylabel('Counts')
    plt.xlabel('outputs (nA)')
    plt.savefig(data_directory + '/output_distribution')


def prepare_data(inputs, outputs, clipping_value):

    mean_output = np.mean(outputs, axis=1)
    # Get cropping mask
    if type(clipping_value) is list:
        cropping_mask = (mean_output < clipping_value[1]) * (mean_output > clipping_value[0])
    elif type(clipping_value) is float:
        cropping_mask = np.abs(mean_output) < clipping_value
    else:
        TypeError(f"Clipping value not recognized! Must be list with lower and upper bound or float, was {type(clipping_value)}")

    outputs = outputs[cropping_mask]
    inputs = inputs[cropping_mask, :]
    return inputs, outputs

############################################################################
# TODO:


def data_merger(list_dirs):
    NotImplementedError('Merging of data from a list of data directories not implemented!')
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


if __name__ == '__main__':
    data_directory = "C:/Users/NE-admin/Documents/Brainspy/brainspy-smg/tmp/data/training/Brains_2020_03_14_203243"
    # The post_process function should have a clipping value which is in an amplified scale.
    # E.g., for an amplitude of 100
    inputs, outputs, info = post_process(data_directory, clipping_value=[-110, 110])
    output_hist(outputs, data_directory, bins=500)
    plt.show()
