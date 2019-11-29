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


def post_process(data_directory, threshold=[-np.inf, np.inf], **kwargs):
    '''Postprocess data, cleans clipping, and merges data sets if needed. The data arrays are merged
    into a single array and cropped given the thresholds. The function also plots and saves the histogram of the data
    Arguments:
        - a string with path to the directory with the data: it is assumed there is a
        sampler_configs.json and a IO.dat file.
        - threshold (kwarg): A lower and upper threshold to crop data; default is [-np.inf,np.inf]
    Optiponal kwargs:
        - list_data: A list of strings indicating directories with training_NN_data.npz containing 'data'.

    NOTE:
        - The data is saved in path_to_data to a .npz file with keyes: inputs, outputs and info_dict,
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

    # Get reference batches
    reference_inputs = inputs[:2 * configs['input_data']['batch_points']]
    reference_outputs = outputs[:2 * configs['input_data']['batch_points']]
    save_npz(data_directory, 'reference_signal', reference_inputs, reference_outputs, configs)
    # Plot histogram save
    output_hist(outputs, data_directory, bins=500)
    # Clean data
    inputs, outputs = prepare_data(inputs, outputs, threshold)
    # save data
    save_npz(data_directory, 'training_data', inputs, outputs, configs)


def save_npz(data_directory, file_name, inputs, outputs, configs):
    save_to = os.path.join(data_directory, file_name)
    print(f'Data saved to \n {save_to}')
    np.savez(save_to, inputs=inputs, outputs=outputs, info=configs)


def output_hist(outputs, data_directory, bins=500):
    plt.figure()
    plt.suptitle('Output Histogram')
    plt.hist(outputs)
    plt.ylabel('Counts')
    plt.xlabel('outputs (nA)')
    plt.savefig(data_directory + '/output_distribution')


def prepare_data(inputs, outputs, threshold):

    nr_raw_samples = len(outputs)
    print('Number of raw samples: ', nr_raw_samples)
    mean_output = np.mean(outputs, axis=1)
    # Get cropping mask
    if type(threshold) is list:
        cropping_mask = (mean_output < threshold[1]) * (mean_output > threshold[0])
    elif type(threshold) is float:
        cropping_mask = np.abs(mean_output) < threshold
    else:
        assert False, "Threshold not recognized! Must be list with lower and upper bound or float."

    outputs = outputs[cropping_mask]
    inputs = inputs[cropping_mask, :]
    print('% of points cropped: ', (1 - len(outputs) / nr_raw_samples) * 100)


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
    data_directory = "tmp\\data\\TEST\\toy_data_2019_11_28_162803\\IO.dat"
    post_process(data_directory)
