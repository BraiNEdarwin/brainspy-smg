# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018
Script to sample a device using waves
@author: M. Boon & HC Ruiz
"""

# import save
# import instruments
import time
import numpy as np


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

    rads = (2 * np.pi * configs['freq'][:, np.newaxis] * points) / configs['sample_frequency'] + configs['phase'][:, np.newaxis]

    if configs['wave_type'] is 'sine':
        wave = np.sin(rads)
    elif configs['wave_type'] is 'triangle':
        # There is an additional + np.pi/2 to make sure that if phase = 0. the inputs start at 0V
        wave = signal.sawtooth(rads + np.pi / 2, width=0.5)

    return wave * configs['amplitude'][:, np.newaxis] + np.outer(configs['offset'], np.ones(time_points.shape[0]))

# initialize save directory


def generate_triangle(self, freq, t, amplitude, fs, phase=np.zeros(7)):
    '''
    Generates a triangle wave form that can be used for the input data.
    freq:       Frequencies of the inputs in an one-dimensional array
    t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
    amplitude:  Amplitude of the sine wave (Vmax in this case)
    fs:         Sample frequency of the device
    phase:      (Optional) phase offset at t=0
    '''
    # There is an additional + np.pi/2 to make sure that if phase = 0. the inputs start at 0V

    return signal.sawtooth((2 * np.pi * freq[:, np.newaxis] * t) / fs + phase[:, np.newaxis] + np.pi / 2, 0.5) * amplitude[:, np.newaxis]


def generate_sinewave(self, freq, t, amplitude, fs, phase=np.zeros(7)):
    '''
    Generates a sine wave that can be used for the input data.
    freq:       Frequencies of the inputs in an one-dimensional array
    t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
    amplitude:  Amplitude of the sine wave (Vmax in this case)
    fs:         Sample frequency of the device
    phase:      (Optional) phase offset at t=0
    '''

    return np.sin((2 * np.pi * freq[:, np.newaxis] * t) / fs + phase[:, np.newaxis]) * amplitude[:, np.newaxis]


def wave_sampler(configs):
    # Initialize output containers
    data = np.zeros((int(configs['sampleTime'] * configs['sample_frequency']), 1))

    nr_batches = int(configs['sample_frequency'] * configs['sampleTime'] / configs['samplePoints'])

    for i in range(batches):
        start_batch = time.time()

        time_points = np.arange(i * configs['samplePoints'], (i + 1) * configs['samplePoints'])
        waves = input_generator(time_points, configs)
        # Use 0.5 second to ramp up to the value where data aqcuisition stopped previous iteration
        # and 0.5 second to ramp down after the batch is done
        wavesRamped = np.zeros((waves.shape[0], waves.shape[1] + int(configs['sample_frequency'])))
        dataRamped = np.zeros(wavesRamped.shape[1])
        for j in range(wavesRamped.shape[0]):
            wavesRamped[j, 0:int(0.5 * configs['sample_frequency'])] = np.linspace(0, waves[j, 0], int(0.5 * configs['sample_frequency']))
            wavesRamped[j, int(0.5 * configs['sample_frequency']): int(0.5 * configs['sample_frequency']) + waves.shape[1]] = waves[j, :]
            wavesRamped[j, int(0.5 * configs['sample_frequency']) + waves.shape[1]:] = np.linspace(waves[j, -1], 0, int(0.5 * configs['sample_frequency']))

        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, configs['sample_frequency'])
        data[i * configs['samplePoints']: (i + 1) * configs['samplePoints'], 0] = dataRamped[0, int(0.5 * configs['sample_frequency']):int(0.5 * configs['sample_frequency']) + waves.shape[1]]

    if i % 10 == 0:  # Save after every 10 batches
        print('Saving...')
        SaveLib.saveExperiment(configs['configSrc, saveDirectory,
                               output=data * configs['amplification'] / configs['postgain'],
                               freq=configs['freq'],
                               sampleTime=configs['sampleTime'],
                               sample_frequency=configs['sample_frequency'],
                               phase=configs['phase'],
                               amplitude=configs['amplitude'],
                               offset=configs['offset'],
                               amplification=configs['amplification'],
                               electrodeSetup=configs['electrodeSetup'],
                               gain_info=configs['gain_info'],
                               filename='training_NN_data')
    end_batch= time.time()
    print('Data collection for part ' + str(i + 1) + ' of ' + str(batches) + ' took ' + str(end_batch - start_batch) + ' sec.')

SaveLib.saveExperiment(configs['configSrc, saveDirectory,
                        output=data * configs['amplification / configs['postgain,
                        freq=configs['freq,
                        sampleTime=configs['sampleTime,
                        sample_frequency=configs['sample_frequency,
                        phase=configs['phase,
                        amplitude=configs['amplitude,
                        offset=configs['offset,
                        amplification=configs['amplification,
                        electrodeSetup=configs['electrodeSetup,
                        gain_info=configs['gain_info,
                        filename='training_NN_data')
