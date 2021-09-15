import numpy as np
from scipy import signal


def get_input_generator(configs):
    if configs["input_data"]["input_distribution"] == "sine":
        return load_configs(configs), sine_wave
    elif configs["input_data"]["input_distribution"] == "sawtooth":
        return load_configs(configs), sawtooth_wave
    # elif configs["input_data"]["input_distribution"] == "uniform_random":
    #     raise NotImplementedError(
    #         'Uniform random wave generator not available')
    else:
        raise NotImplementedError(
            f"Input wave array type {configs['input_distribution']} not recognized"
        )


def sine_wave(time_points: int, frequency: int, phase: float, amplitude: float,
              offset: float):
    """
    Generates a sine wave.

    Parameters
    ----------
    time_points : int
        Time points to evaluate the function.
    frequency : int
        Frequencies of the inputs.
    phase : float
        Phase offset of sine wave.
    amplitude : float
        Amplitude of the sine wave.
    offset : float
        Offset of the input.

    Returns
    -------
    np.array
        Sine wave.
    """
    return amplitude * np.sin(2 * np.pi * frequency * time_points +
                              phase) + np.outer(offset,
                                                np.ones(len(time_points)))


def sawtooth_wave(time_points: int, frequency: int, phase: float,
                  amplitude: float, offset: float):
    """
    Generates a sawtooth wave.

    Parameters
    ----------
    time_points : int
        Time points to evaluate the function.
    frequency : int
        Frequencies of the inputs.
    phase : float
        Phase offset of wave.
    amplitude : float
        Amplitude of the wave.
    offset : float
        Offset of the input.

    Returns
    -------
    np.array
        Sawtooth wave.
    """
    rads = 2 * np.pi * frequency * time_points + phase
    wave = signal.sawtooth(rads + np.pi / 2, width=0.5)
    return amplitude * wave + np.outer(offset, np.ones(len(time_points)))


# def uniform_random_wave(configs):
#     '''
#     Generates a waveform with random amplitudes
#     Args:
#         configs: Dictionary containing all the sampling configurations, including
#                     sample_frequency: Sample frequency of the device
#                     length: length of the amplitudes
#                     slope: slope between two amplitudes
#     '''
#     raise NotImplementedError('Uniform random waveform not implemented')


def load_configs(config_dict):
    configs = config_dict["input_data"]
    configs['sampling_frequency'] = config_dict["driver"]['sampling_frequency']
    configs['input_frequency'] = get_frequency(configs)
    configs['phase'] = np.array(configs['phase'])[:, np.newaxis]
    configs['amplitude'] = configs['amplitude'][:, np.newaxis]
    configs['offset'] = configs['offset'][:, np.newaxis]
    configs[
        'batch_points'] = configs['batch_time'] * configs['sampling_frequency']
    configs[
        'ramp_points'] = configs['ramp_time'] * configs['sampling_frequency']
    return configs


def get_frequency(configs):
    aux = np.array(configs['input_frequency'])[:, np.newaxis]
    return np.sqrt(
        aux[:configs['activation_electrode_no']]) * configs['factor']


def generate_sawtooth_multiple(input_range, n_points, direction):

    n_points = n_points / 2

    if direction == "up":
        Input1 = np.linspace(
            0, input_range[0],
            int((n_points * input_range[0]) /
                (input_range[0] - input_range[1])))
        Input2 = np.linspace(input_range[0], input_range[1], int(n_points))
        Input3 = np.linspace(
            input_range[1], 0,
            int((n_points * input_range[1]) /
                (input_range[1] - input_range[0])))
    elif direction == "down":
        Input1 = np.linspace(
            0, input_range[1],
            int((n_points * input_range[1]) /
                (input_range[1] - input_range[0])))
        Input2 = np.linspace(input_range[1], input_range[0], int(n_points))
        Input3 = np.linspace(
            input_range[0], 0,
            int((n_points * input_range[0]) /
                (input_range[0] - input_range[1])))
    else:
        print('Specify the sweep direction')
    result = np.concatenate((Input1, Input2, Input3))
    if not (result.shape[0] == int(n_points * 2)):
        result = np.concatenate((result, np.array([0])))
    return result


def generate_sawtooth_simple(v_low: float,
                             v_high: float,
                             point_no: int,
                             up_direction: bool = False):
    """Generates a simple sawtooth for a single channel (electrode). It goes from zero to a certain
    point (v_low), from that point to another point (v_max), and from that last point to zero again.
    The direction can be inverted using up_direction=True so that the sawtooth goes from zero to 
    v_max, from v_max to v_min, and from v_min to zero.

    Parameters
    ----------
    v_low : float
        Minimum voltage that the sawtooth will achieve.
    v_high : float
        Maximum voltage that the sawtooth will achieve.
    point_no : int
        Number of points that the sawtooth will have.
    up_direction : bool, optional
        Direction of the sawtooth. If true, the sawtooth will go first up and then down. 
        If False, the sawtooth will go first down and then up. By default False.

    Returns
    -------
    np.array
        An array containing the two pointed sawtooth in a single dimension.
    """
    assert point_no % 2 == 0, 'Only an even point number is accepted.'
    point_no = int(point_no / 2)

    if up_direction:
        aux = v_low
        v_low = v_high
        v_high = aux

    ramp1 = np.linspace(0, v_low, int((point_no * v_low) / (v_low - v_high)))
    ramp2 = np.linspace(v_low, v_high, point_no)
    ramp3 = np.linspace(v_high, 0, int((point_no * v_high) / (v_high - v_low)))

    result = np.concatenate((ramp1, ramp2, ramp3))
    return result


def generate_sinewave(n, fs, amplitude, phase=0):
    '''
	Generates a sine wave that can be used for the input data.
	freq:       Frequencies of the inputs in an one-dimensional array
	t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
	amplitude:  Amplitude of the sine wave (Vmax in this case)
	fs:         Sample frequency of the device
	phase:      (Optional) phase offset at t=0
	'''
    freq = fs / n
    points = np.linspace(0, 1 / freq, n)
    phases = points * 2 * np.pi * freq

    return np.sin(phases + phase) * amplitude
