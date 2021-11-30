import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager
from itertools import cycle
import pickle

def generate_inputs(ranges, data_input_indices, waveform_mgr: WaveformManager):
    ranges = TorchUtils.format(ranges).double()
    result = torch.zeros_like(ranges).double()
    result[data_input_indices] = ranges[data_input_indices].clone()
    result = waveform_mgr.points_to_waveform(result.T)
    mask = TorchUtils.to_numpy(waveform_mgr.generate_mask(result.shape[0]))
    return TorchUtils.to_numpy(result), mask
    
def plot_inputs(inputs, save_dir):
    plt.figure()
    for i in range(inputs.shape[1]):
        plt.plot(inputs[:, i], label=f'Electrode {i}', alpha=0.5)
    plt.legend()
    plt.title('Inputs to the device')
    plt.savefig(os.path.join(save_dir, 'inputs.png'))
    plt.close()

def plot_outputs(outputs, frequencies, data_dir=None, mask=None):
    plt.figure()
    color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])
    for i in range(outputs.shape[1]):
        color = next(color_cycle)
        if mask is None:
            mean = outputs[:, i].mean(axis=0)
            std = outputs[:, i].std(axis=0)
           
        else:
            mean = outputs[:, i].T[mask].T.mean(axis=0)
            std = outputs[:, i].T[mask].T.std(axis=0)
            #plt.plot(outputs[i][mask], label=f'Frequency {frequencies[i]} Hz', alpha=0.5)
        plt.plot(mean, label=f'Frequency {frequencies[i]} Hz', c=color)
        plt.plot(mean - std, alpha=0.5, linestyle='dashed', c=color)
        plt.plot(mean + std, label=f'Std {frequencies[i]} Hz', alpha=0.5, linestyle='dashed', c=color)
        plt.legend()
    if mask is None:
        plt.title('Outputs in different frequencies')
        if data_dir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(data_dir, 'outputs.png'))
    else:
        plt.title('Outputs in different frequencies (masked ramps)')
        if data_dir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(data_dir, 'outputs_masked.png'))

def plot_results(results_dir):
    results = np.load(os.path.join(results_dir, 'results.npz'))
    plot_inputs(results['inputs'])
    plot_outputs(results['outputs'],results['frequencies'], mask=results['mask'])

if __name__ == '__main__':
    from brainspy.utils.io import load_configs, create_directory_timestamp
    from brainspy.utils.manager import get_driver
    
    configs = load_configs('configs/utils/brains_ivcurve_template.yaml')
    
    
    waveform_mgr = WaveformManager(configs['waveform'])

    data_dir = create_directory_timestamp(configs['save_dir'], configs['test_name'])

    inputs, mask = generate_inputs(configs['driver']['instruments_setup']['activation_voltage_ranges'], configs['data_input_indices'], waveform_mgr)
    
    plot_inputs(inputs, data_dir)
    all_outputs = np.zeros((configs['repetitions'], len(configs['frequencies']), inputs.shape[0]))

    for i in range(len(configs['frequencies'])):
        for j in range(configs['repetitions']):
            print('Repetition: '+ str(j))
            configs['driver']['sampling_frequency'] = configs['frequencies'][i]
            driver = get_driver(configs["driver"])
            output = driver.forward_numpy(inputs)
            driver.close_tasks()
            all_outputs[j, i] = output[:, 0].copy()  # It should be changed for allowing multiple outputs
        # print('')
    np.savez(os.path.join(data_dir, 'results'), inputs=inputs, outputs=all_outputs, mask=mask, frequencies=configs['frequencies'])
    plot_outputs(all_outputs, configs['frequencies'], data_dir)
    plot_outputs(all_outputs, configs['frequencies'], data_dir, mask=mask)
    plt.show()
    plt.close()
    print("Plots saved in: " + data_dir)
