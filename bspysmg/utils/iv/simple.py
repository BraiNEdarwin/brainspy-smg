import numpy as np
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver
from bspysmg.utils.plots import iv_plot
from bspysmg.utils.inputs import generate_sawtooth_simple


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
                 save_plot=None):
        activation_electrode_no = len(
            configs['instruments_setup']['activation_channels'])
        data = np.zeros((activation_electrode_no, point_no))
        data[input_electrode] = generate_sawtooth_simple(
            vmax, vmin, point_no, up_direction)
        result = self.driver.forward_numpy(data.T)
        if close:
            self.driver.close_tasks()
        iv_plot(result, input_electrode, save_plot, show_plot)
        return result


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    configs = load_configs('configs/utils/brains_ivcurve_template_simple.yaml')
    measurement = IVMeasurement(configs)
    for i in range(7):
        measurement.iv_curve(-0.08, 0.08, point_no=5000, input_electrode=i, show_plot=True, close=False)
    measurement.driver.close_tasks()
    
