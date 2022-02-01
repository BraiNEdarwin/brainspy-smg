import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def generate_inputs(configs):
    inputs = np.zeros((7, configs['shape']))
    assert configs['source_electrode_index'] != configs['gate_electrode_index'], 'Gate and Source electodes cannot be the same!'
    
    Vsd = 0.2
    assert Vsd < 0.6, 'DANGER!! Reduce Source-Drain voltage'

    slope_points = 100
    ramp1 = np.linspace(0, Vsd, slope_points)
    plateau = np.linspace(Vsd, Vsd, configs['shape'] - 2 * slope_points)
    ramp2 = np.linspace(Vsd, 0, slope_points)
    
    inputs[configs['source_electrode_index'],:] = np.concatenate((ramp1, plateau, ramp2))

    return inputs
    
def plot_inputs(inputs):
    plt.figure()
    for i in range(inputs.shape[0]):
        plt.plot(inputs[i, :],
                label = f'Electrode {i}', alpha = 0.5)
    plt.legend()
    plt.title('Input spikes to the device')
    plt.show()
    plt.close()

def plot_output(output, inputs,  configs):
    plt.figure()
    if inputs != None and configs != None:  
        plt.plot(inputs[configs['source_electrode_index'], :],
                output,
                label = f'Output response from device', alpha = 0.5)
    else:
        plt.plot(output)
    plt.legend()
    plt.title('Output response from device')
    plt.show()
    plt.close()

def plot_psd(output):
    plt.figure()
    #plt.psd(output, NFFT=512, Fs=1000, window=mlab.window_none, pad_to=2048, noverlap=128, scale_by_freq=True)
    # pxx, freqs = plt.psd(output, NFFT=512, Fs=1000, window=mlab.window_none, pad_to=2048, noverlap=128, scale_by_freq=True)
    pxx, freqs = plt.psd(output[:,0], Fs=512, window=mlab.window_none, noverlap=128, scale_by_freq=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.plot(freqs, pxx)
    plt.show()
    plt.close()

if __name__ == '__main__':
    from brainspy.utils.manager import get_driver
    from brainspy.utils.io import load_configs

    configs = load_configs('configs/utils/brains_ivcurve_template_simple.yaml')
    configs['input_sine_frequency'] = 1 # Freq. in Hz
    configs['source_electrode_index'] = 5
    configs['gate_electrode_index'] = 6


    inputs_to_device = generate_inputs(configs)

    plot_inputs(inputs_to_device)

    print("Total measurement time: ", configs['shape']/configs['driver']['instruments_setup']['activation_sampling_frequency'], "(s)")

    driver = get_driver(configs["driver"])
    output = driver.forward_numpy(inputs_to_device.T)
    driver.close_tasks()

    plot_output(output, inputs=None, configs= None)
    plot_psd(output[50:9950])
    plt.show()
    plt.close()


