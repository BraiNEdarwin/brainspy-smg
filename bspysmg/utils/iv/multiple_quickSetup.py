import numpy as np
from brainspy.utils.manager import get_driver

from bspysmg.utils.inputs import generate_sawtooth_simple

import matplotlib.pyplot as plt

import torch
from brainspy.utils.waveform import WaveformManager


class MultiIVMeasurement():
    def __init__(self, configs) -> None:
        self.configs = configs
        self.input_signal = self.configs['input_signal']
        
    def run_test(self):
        self.driver = get_driver(self.configs['driver'])
        #experiments = ["IV1", "IV2", "IV3", "IV4", "IV5", "IV6", "IV7"]
        outputs = {}
        inputs = {}
        for exp in range(7): # self.configs['driver']['instruments_setup']['activation_channels']: #range(len(experiments)):
            inputs[exp] = self.create_input_arrays(exp)
            result = self.driver.forward_numpy(
                inputs[exp]
            )
            outputs[exp] = result

        self.driver.close_tasks()
        self.multi_iv_plot(inputs, outputs)

    def create_input_arrays(self, exp):
        input_array = np.zeros(
            (self.configs['driver']['instruments_setup']['activation_channel_mask'].count(1),
             self.configs['shape'])
        )
        input_array[exp,:] = self.gen_input_waveform(
            self.configs['driver']['instruments_setup'][
                'activation_voltage_ranges'][exp]
        )
    
        return input_array.T

    def gen_input_waveform(self, input_range): #only works for sawtooth
        if self.input_signal['input_signal_type'] == 'sawtooth':
            input_data = generate_sawtooth_simple(
                input_range[0], input_range[1],
                self.configs['shape'],
                self.input_signal['direction']
            )
        else:
            print("For now, it only works for sawtooth")
        
        return np.array(input_data)

    def multi_iv_plot(self, inputs, outputs):
        ylabeldist = -10
        electrode_id = 0
        cmap = plt.get_cmap("tab10")
        fig, axs = plt.subplots(2, 4)
        fig.suptitle('Input voltage vs Output current')
        for i in range(2):
            for j in range(4):
                exp_index = j + i * 4
                exp = "IV" + str(exp_index + 1)
                if exp_index < 7:
                    wm = WaveformManager({'slope_length':0, 'plateau_length': int(10000/self.configs['driver']['DAC_update_rate'])})
                    y_axis = wm.points_to_plateaus(torch.tensor(inputs[exp_index][:,exp_index])).detach().cpu().numpy()#[int(10000/self.configs['driver']['DAC_update_rate']):]
                    axs[i, j].plot(y_axis, outputs[exp_index],
                    color=cmap(exp_index))
                    axs[i, j].set_ylabel('output (nA)',
                    labelpad=ylabeldist)
                    axs[i, j].set_xlabel('input (V)',
                    labelpad=1)
                    axs[i, j].xaxis.grid(True)
                    axs[i, j].yaxis.grid(True)
                else:
                    for m in range(7):
                        wm = WaveformManager({'slope_length':0, 'plateau_length': int(10000/self.configs['driver']['DAC_update_rate'])})
                        y_axis = wm.points_to_plateaus(torch.tensor(inputs[m][:,m])).detach().cpu().numpy()[int(10000/self.configs['driver']['DAC_update_rate']):]
                        axs[i, j].plot(y_axis,
                                        label="IV"+str(m),
                                        color=cmap(m))
                    axs[i, j].set_ylabel('input (V)')
                    axs[i, j].set_xlabel('points', labelpad=1)
                    axs[i, j].set_title("Input input_signal")
                    axs[i, j].xaxis.grid(True)
                    axs[i, j].yaxis.grid(True)
                    axs[i, j].legend()

        plt.subplots_adjust(hspace=0.3, wspace=0.35)
        plt.show()

if __name__ == '__main__':

    from brainspy.utils.io import load_configs

    configs = load_configs('configs/utils/quick_ivcurve.yaml')

    test = MultiIVMeasurement(configs)
    test.run_test()
