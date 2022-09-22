"""
File containing a class for measuring IV curves on multiple devices PCBs.
"""
import numpy as np
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver
from bspysmg.utils.plots import iv_plot
from bspysmg.utils.inputs import generate_sawtooth_simple

from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils
import torch
import collections


class IVMeasurement():
    
    def __init__(self, configs: dict, info: dict = None,
        model_state_dict: collections.OrderedDict = None) -> None:
        """
        Initializes the driver for which IV curve is to be plotted. It uses a config dict to
        initialize the driver. The driver can be the DNPU device itself (on chip training) or
        a surrogate model (off chip training). This class only allows for measurement of IV
        curves of one device at a time.

        Parameters
        ----------
        configs : dict
            ivcurve configurations with the following keys:
            1. processor_type:  str
            The type of driver for which ivcurve is to be measured. It can take following
            values: cdaq_to_cdaq, cdaq_to_nidaq, simulation_debug.
        """
        self.driver = TorchUtils.format(Processor(configs, info, model_state_dict))

    def iv_curve(self,
                 vmax: float,
                 vmin: float,
                 point_no: int,
                 input_electrode: int,
                 up_direction: bool = True,
                 close: bool = True,
                 show_plot: bool = False,
                 save_plot: bool = None) -> np.array:
        """
        Computes the IV response of DNPU device or surrogate model to an input sawtooth signal.
        Optionally shows the graph on screen and saves it to current directory. This is done
        to check if a particular electrode has non linear IV response or a negative differential
        resistance (NDR).
        Also check:
        
        https://en.wikibooks.org/wiki/Circuit_Idea/Negative_Differential_Resistance
        
        https://resources.pcb.cadence.com/blog/2019-what-is-linear-and-nonlinear-resistance

        Parameters
        ----------
        v_max : float
            Maximum voltage that the sawtooth will achieve.
        v_min : float
            Minimum voltage that the sawtooth will achieve.
        point_no : int
            Number of points that the sawtooth will have.
        input_electrode : int
            Electrode number.
        up_direction : bool [Optional]
            Direction of the sawtooth. If true, the sawtooth will go first up and then down.
            If False, the sawtooth will go first down and then up. By default False.
        close : bool [Optional]
            If set to true, it closes the driver and clears it from memory.
        show_plot : bool [Optional]
            If set to true, it displays the generated plot.
        save_plot : bool [Optional]
            If set to true, it saves the generated plot to current directory.
        
        Returns
        ----------
        result : np.array
            IV response of device or surrogate model.
        """
        data = np.zeros((self.driver.get_activation_electrode_no(), point_no))
        data[input_electrode] = generate_sawtooth_simple(
            vmax, vmin, point_no, up_direction)
        result = self.driver(TorchUtils.format(data.T))
        if close:
            self.driver.close()
        return data[input_electrode], result.detach().cpu().numpy()