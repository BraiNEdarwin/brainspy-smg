from matplotlib import colors
import numpy as np

from bspysmg.utils.inputs import generate_sawtooth_simple
import matplotlib.pyplot as plt
from itertools import cycle



def generate_inputs(v_low: float,
                    v_high: float,
                    slope_points,
                    plateau_points
                    ):

    assert (slope_points >= 5), 'DANGER: too few ramp points might damage the device!'
    assert (plateau_points >= 1), 'DANGER: too few plateau points might damage the device!'

    ramp1 = np.linspace(0, 0.8*v_low, slope_points)
    plateau1 = np.linspace(0.8*v_low, 0.8*v_low, plateau_points) + 0.02 * np.sin(
        np.linspace(-2*np.pi,2*np.pi , plateau_points)
    )
    ramp2 = np.linspace(0.8*v_low, 0.8*v_high, slope_points)

    plateau2 = np.linspace(0.8*v_high, 0.8*v_high, plateau_points) + 0.02 * np.sin(
        np.linspace(-2*np.pi,2*np.pi , plateau_points)
    )

    ramp3 = np.linspace(0.8*v_high, 0, slope_points)

    result = np.concatenate((ramp1, plateau1, ramp2, plateau2, ramp3))

    return result

def plot_all(inputs, outputs, frequencies):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Output current vs Input voltage')
    for i in range(inputs.shape[0]):
        axs[0].plot(inputs[i, :],
                    label=f'Electrode {i}',
                    alpha=0.5
        )
    axs[0].set_ylabel('Voltage (V)')
    axs[0].set_xlabel('Points (Unit time)')
    axs[0].legend()
    """Next"""
    color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])
    for i in range(len(outputs)):
        color = next(color_cycle)
        x_axis = np.linspace(0, 1, outputs[i].shape[0])
        axs[1].plot(x_axis,
                    outputs[i],
                    label=f'Frequency {frequencies[i]} Hz',
                    c=color
        )
        # axs[1].plot(outputs[i],
        #              label=f'Frequency {frequencies[i]} Hz',
        #              c=color)
    axs[1].set_ylabel('Current (nA)')
    axs[1].set_xlabel('Normalized points*')
    axs[1].legend()
    plt.show()

def plot_outputs(outputs, frequencies):
    plt.figure()
    color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])
    for i in range(len(outputs)):
        color = next(color_cycle)
        x_axis = np.linspace(0, 1, outputs[i].shape[0])
        plt.plot(x_axis, outputs[i],
                label=f'Frequency {frequencies[i]} Hz',
                c=color
        )
        plt.legend()
    plt.show()

def plot_inputs(inputs):
    plt.figure()
    for i in range(inputs.shape[0]):
        plt.plot(inputs[i, :],
                label= f'Electrode {i}',
                alpha=0.5
        )
    plt.legend()
    plt.title('Inputs to the device')
    plt.show()


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    from brainspy.utils.manager import get_driver
    import time

    configs = load_configs('configs/utils/quick_ivcurve.yaml')

    input_array = np.zeros(
            (configs['driver']['instruments_setup'][
                'activation_channel_mask'].count(1),
             configs['slope_length']*3 + configs['plateau_length']*2)
    )
    
    selected_electrodes = [1]

    for se in selected_electrodes:
        input_array[se,:] = generate_inputs(v_low= configs['driver']['instruments_setup'][
            'activation_voltage_ranges'][se][0],
            v_high= configs['driver']['instruments_setup'][
            'activation_voltage_ranges'][se][1],
            slope_points= configs['slope_length'],
            plateau_points=configs['plateau_length'])

    all_outputs = {}
    for i in range(len(configs['frequencies'])):
        print("DAC Frequency to test: ", configs['frequencies'][i])
        configs['driver']['DAC_update_rate'] = configs['frequencies'][i]
        driver = get_driver(configs['driver'])
        all_outputs[i] = driver.forward_numpy(input_array.T)
        driver.close_tasks()

        #x_axis = np.linspace(0, 1, output.shape[0])

        #plt.plot(x_axis, output)
        #plt.legend()

        time.sleep(2)
    
    #plot_inputs(input_array)
    #plot_outputs(all_outputs, configs['frequencies'])
    plot_all(input_array, all_outputs, configs['frequencies'])
