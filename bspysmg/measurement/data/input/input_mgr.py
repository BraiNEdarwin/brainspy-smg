# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018
Script to sample a device using waves
@author: HC Ruiz
"""

import numpy as np
from scipy import signal


def get_input_generator(configs):
    if configs["input_distribution"] == "sine":
        return load_configs(configs), sine_wave
    elif configs["input_distribution"] == "sawtooth":
        return load_configs(configs), sawtooth_wave
    elif configs["input_distribution"] == "uniform_random":
        raise NotImplementedError(f'Uniform random wave generator not available')
    else:
        raise NotImplementedError(f"Input wave array type {configs['input_distribution']} not recognized")


def complete_sine(configs):
    '''Args:
        configs: Dictionary containing all the sampling wave configurations
    '''
    all_time_points = np.arange(configs["batch_points"]) / configs["sampling_frequency"]
    return sine_wave(all_time_points, configs['input_frequency'], configs['phase'], configs['amplitude'], configs['offset'])


def complete_sawtooth(configs):
    '''Args:
    configs: Dictionary containing all the sampling wave configurations
    '''
    all_time_points = np.arange(configs["batch_points"]) / configs["sampling_frequency"]
    return sawtooth_wave(all_time_points, configs['input_frequency'], configs['phase'], configs['amplitude'], configs['offset'])

###################################
#         WAVE GENERATORS         #
###################################


def sine_wave(time_points, frequency, phase, amplitude, offset):
    '''
    Generates a sine wave
    Arguments:
        time_points: Time points to evaluate the function
        frequency:   Frequencies of the inputs 
        amplitude:   Amplitude of the sine wave 
        phase:       Phase offset of sine wave 
        offset:      Offset of the input
    '''
    return amplitude * np.sin(2 * np.pi * input_frequency * time_points + phase) + np.outer(offset, np.ones(len(time_points)))


def sawtooth_wave(time_points, frequency, phase, amplitude, offset):
    '''
    Generates a sawtooth wave
    Arguments:
        time_points: Time points to evaluate the function
        frequency:   Frequencies of the inputs 
        amplitude:   Amplitude of the wave 
        phase:       Phase offset of wave 
        offset:      Offset of the input
    '''
    rads = 2 * np.pi * frequency * time_points + phase
    wave = signal.sawtooth(rads + np.pi / 2, width=0.5)
    return amplitude * wave + np.outer(offset, np.ones(len(time_points)))


def uniform_random_wave(configs):
    '''
    Generates a waveform with random amplitudes
    Args:
        configs: Dictionary containing all the sampling configurations, including
                    sample_frequency: Sample frequency of the device
                    length: length of the amplitudes
                    slope: slope between two amplitudes
    '''
    raise NotImplementedError('Uniform random waveform not implemented')

######################
#  HELPER FUNCTIONS  #
######################


def load_configs(configs):
    configs['input_frequency'] = get_frequency(configs)
    configs['phase'] = np.array(configs['phase'])[:, np.newaxis]
    configs['amplitude'] = np.array(configs['amplitude'])[:, np.newaxis]
    configs['offset'] = np.array(configs['offset'])[:, np.newaxis]
    configs['batch_points'] = configs['batch_time'] * configs['sampling_frequency']
    configs['ramp_points'] = configs['ramp_time'] * configs['sampling_frequency']
    return configs


def get_frequency(configs):
    aux = np.array(configs['input_frequency'])[:, np.newaxis]
    return np.sqrt(aux[:configs['input_electrodes']]) * configs['factor']
