import numpy as np
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver
from bspysmg.utils.plots import iv_plot
from bspysmg.utils.inputs import generate_sawtooth_simple
import matplotlib.pyplot as plt
import torch


class IVMeasurement():
    def __init__(self, configs):
        self.driver = get_driver(configs)

    def iv_curve(self,
                 vmax: float,
                 vmin: float,
                 point_no: int,
                 input_electrode: int,
                 up_direction: bool = True,
                 close: bool = True,
                 show_plot=False,
                 save_plot=None,
                 sampling_frequency = 5000):
        wm = WaveformManager({'slope_length':0, 'plateau_length': int(10000/sampling_frequency)})
        activation_electrode_no = len(
            configs['instruments_setup']['activation_channels'])
        data = np.zeros((activation_electrode_no, point_no))
        data[input_electrode] = generate_sawtooth_simple(
            vmax, vmin, point_no, up_direction)
        result = self.driver.forward_numpy(data.T)
        if close:
            self.driver.close_tasks()
        iv_plot(result, wm.points_to_plateaus(torch.tensor(data[input_electrode])).detach().cpu().numpy()[int(10000/sampling_frequency):], input_electrode, save_plot, show_plot)
        return result

    def iv_NDR(self,
                input_vmax: float,
                input_vmin: float,
                cv_voltage: float,
                point_no: int,
                input_electrode: int,
                cv_electrode: int,
                up_direction: bool = True,
                close: bool = True,
                show_plot: bool = True,
                save_plot = None,
                sampling_frequency = 100,
                slope_length = 20
            ):
        wm = WaveformManager({'slope_length':0, 'plateau_length': int(10000/sampling_frequency)})
        activation_electrode_no = len(configs['instruments_setup']['activation_channels'])
        data = np.zeros((activation_electrode_no,
                         point_no + int(slope_length))) 
        data[input_electrode][int(slope_length/2):point_no+int(slope_length/2)] = generate_sawtooth_simple(
            input_vmax, input_vmin, point_no, up_direction
        )
        data[cv_electrode] = np.concatenate((
                np.linspace(0, cv_voltage, int(slope_length/2)), np.linspace(cv_voltage, cv_voltage, point_no),
                np.linspace(cv_voltage, 0, int(slope_length/2))
            ))
        result = self.driver.forward_numpy(data.T)
        if close:
            self.driver.close_tasks()
        iv_plot(result, wm.points_to_plateaus(torch.tensor(data[input_electrode])).detach().cpu().numpy()[int(10000/sampling_frequency):], input_electrode, save_plot, show_plot)
        # plt.plot(data[cv_electrode])
        # plt.show()


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    configs = load_configs('configs/utils/brains_ivcurve_template_simple.yaml')
    from brainspy.utils.waveform import WaveformManager

   
    measurement = IVMeasurement(configs)
    #for i in range(7):
    #measurement.iv_curve(-0.4, 0.6, point_no=50, input_electrode=0, show_plot=True, close=True, sampling_frequency=configs['sampling_frequency'])
    measurement.iv_NDR(0.7, -0.7, 0.7, point_no=500, input_electrode=3, cv_electrode=4,
                        sampling_frequency=configs['sampling_frequency'],
                        slope_length=40
    )
    measurement.driver.close_tasks()
    
