# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018
Script to sample a device using waves
@author: M. Boon,  HC Ruiz, & Unai Alegre
"""

import numpy as np
from scipy import signal


def load_configs(configs):
    configs['freq'] = get_frequency(configs['input_frequency'])
    configs['phase'] = np.zeros(configs['input_frequency']['wave_electrodes'])
    return configs


def get_frequency(configs):
    aux = np.array(configs['frequecy_base'])
    return np.sqrt(aux[:configs['wave_electrodes']]) * configs['factor']


def input_generator(points, configs):
    '''
    Generates a wave that can be used as input data.
    Args:
        points: The datapoint(s) index for the wave value (1D array when multiple datapoints are used)
        configs: Dictionary containing all the sampling configurations, including
                    freq:       Frequencies of the inputs in an one-dimensional array
                    amplitude:  Amplitude of the sine wave (Vmax in this case)
                    sample_frequency:         Sample frequency of the device
                    phase:      (Optional) phase offset at t=0
    '''
    configs = load_configs(configs)

    rads = (2 * np.pi * configs['freq'][:, np.newaxis] * points) / configs['sampling_frequency'] + configs['phase'][:, np.newaxis]

    if configs['wave_type'] == 'sine':
        wave = np.sin(rads)
    elif configs['wave_type'] == 'triangle':
        # There is an additional + np.pi/2 to make sure that if phase = 0. the inputs start at 0V
        wave = signal.sawtooth(rads + np.pi / 2, width=0.5)
    else:
        raise NotImplementedError(f"The wave type {configs['wave_type']} is not recognised. Please try again with 'sine' or 'triangle'.")

    return wave * configs['amplitude'][:, np.newaxis] + np.outer(configs['offset'], np.ones(points.shape[0]))


def ramp_signal(waves, configs):
    ramp = int(configs['ramp_time'] * configs['sample_frequency'])
    # Use configs['ramp_time'] second to ramp up to the value where data aqcuisition stopped previous iteration
    # and configs['ramp_time'] second to ramp down after the batch is done
    waves_ramped = np.zeros((waves.shape[0], waves.shape[1] + int(configs['sample_frequency'])))
    for j in range(waves_ramped.shape[0]):
        waves_ramped[j, 0:ramp] = np.linspace(0, waves[j, 0], ramp)
        waves_ramped[j, ramp: ramp + waves.shape[1]] = waves[j, :]
        waves_ramped[j, ramp + waves.shape[1]:] = np.linspace(waves[j, -1], 0, ramp)

    return waves_ramped


def ramped_input_generator(points, configs):
    return ramp_signal(input_generator(points, configs), configs)
