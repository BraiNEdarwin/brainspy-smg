import numpy as np
from scipy import signal
from typing import Tuple, Callable


def get_input_generator(configs: dict) -> Tuple[dict, Callable]:
    """
    Returns the configurations dictionary for generating input signal wave
    and also returns the function that will be used for generating the signal.

    Parameters
    ----------
    configs: dict
        Sampling configurations, the dictionary has the following keys:
        * input_data : dict
            Dictionary containing the information necessary to create the input sampling data.
            - input_distribution: str
                It determines the wave shape of the input. Two main options availeble 'sawtooth'
                and 'sine'. The first option will create saw-like signals, and the second
                sine-wave signals. Sawtooth signals have more coverage on the edges of the
                input range.
            - activation_electrode_no: int
                Number of activation electrodes in the device that wants to be sampled.
            - readout_electrode_no : int
                Number of readout electrodes in the device that wants to be sampled.
            - input_frequency: list
                Base frequencies of the input waves that will be created. In order to optimise
                coverage, irrational numbers are recommended. The list should have the same
                length as the activation electrode number. The input frequency list will be
                square rooted. E.g., for 7 activation electrodes:
                input_frequency = [2, 3, 5, 7, 13, 17, 19]
            - phase : float
                Horizontal shift of the input signals. It is recommended to have random numbers
                which are different for the training, validation and test datasets. These
                numbers will be square rooted and multiplied by a given factor.
            - factor : float
                Given factor by which the input frequencies will be multiplied after square
                rooting them.
            - amplitude : Optional[list[float]]
                Amplitude of the generated input wave signal. It is calculated according to the
                minimum and maximum ranges of each electrode. Where the amplitude value should
                correspond with (max_range_value - min_range_value) / 2. If no amplitude is
                given it will be automatically calculated from the driver configurations for
                activation electrode ranges. If it wants to be manually set, the offset
                variable should also be included in the dictionary.
            - offset: Optional[list[float]]
                Vertical offset of the generated input wave signal. It is calculated according
                to the minimum and maximum ranges of each electrode. Where the offset value
                should correspond with (max_range_value + min_range_value) / 2. If no offset
                is given it will be automatically calculated from the driver configurations for
                activation electrode ranges. If it wants to be manually set, the offset
                variable should also be included in the dictionary.
            - ramp_time: float
                Time that will be taken before sending each batch to go from zero to the first
                point of the batch and to zero from the last point of the batch.
            - batch_time:
                Time that the sampling of each batch will take.
            - number_batches: int
                Number of batches that will be sampled. A default value of 3880 is reccommended.

    Returns
    ----------
    tuple
        Sampling configuration dictionary and signal generating function.
    """
    if configs["input_data"]["input_distribution"] == "sine":
        return load_configs(configs), sine_wave
    elif configs["input_data"]["input_distribution"] == "sawtooth":
        return load_configs(configs), sawtooth_wave
    # elif configs["input_data"]["input_distribution"] == "uniform_random":
    #     raise NotImplementedError(
    #         'Uniform random wave generator not available')
    else:
        raise NotImplementedError(
            f"Input wave array type {configs['input_data']['input_distribution']} not recognized"
        )


def sine_wave(time_points: np.array, frequency: float, phase: float,
              amplitude: float, offset: float) -> np.array:
    """
    Generates a sine wave.

    Parameters
    ----------
    time_points : np.array
        Time points to evaluate the function.
    frequency : float
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


def sawtooth_wave(time_points: np.array, frequency: float, phase: float,
                  amplitude: float, offset: float) -> np.array:
    """
    Generates a sawtooth wave.

    Parameters
    ----------
    time_points : np.array
        Time points to evaluate the function.
    frequency : float
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


def load_configs(config_dict: dict) -> dict:
    """
    Creates a dictionary with sampling configurations.
    Parameters
    ----------
    configs: dict
        Sampling configurations, the dictionary has the following keys:
        * driver: dict
            Dictionary containing the driver configurations. For more information check the
            documentation about this configuration file, check the documentation of
            brainspy.processors.hardware.drivers.ni.setup.NationalInstrumentsSetup
        * input_data : dict
            Dictionary containing the information necessary to create the input sampling data.
            - input_distribution: str
                It determines the wave shape of the input. Two main options availeble 'sawtooth'
                and 'sine'. The first option will create saw-like signals, and the second
                sine-wave signals. Sawtooth signals have more coverage on the edges of the
                input range.
            - activation_electrode_no: int
                Number of activation electrodes in the device that wants to be sampled.
            - readout_electrode_no : int
                Number of readout electrodes in the device that wants to be sampled.
            - input_frequency: list
                Base frequencies of the input waves that will be created. In order to optimise
                coverage, irrational numbers are recommended. The list should have the same
                length as the activation electrode number. The input frequency list will be
                square rooted. E.g., for 7 activation electrodes:
                input_frequency = [2, 3, 5, 7, 13, 17, 19]
            - phase : float
                Horizontal shift of the input signals. It is recommended to have random numbers
                which are different for the training, validation and test datasets. These
                numbers will be square rooted and multiplied by a given factor.
            - factor : float
                Given factor by which the input frequencies will be multiplied after square
                rooting them.
            - amplitude : Optional[list[float]]
                Amplitude of the generated input wave signal. It is calculated according to the
                minimum and maximum ranges of each electrode. Where the amplitude value should
                correspond with (max_range_value - min_range_value) / 2. If no amplitude is
                given it will be automatically calculated from the driver configurations for
                activation electrode ranges. If it wants to be manually set, the offset
                variable should also be included in the dictionary.
            - offset: Optional[list[float]]
                Vertical offset of the generated input wave signal. It is calculated according
                to the minimum and maximum ranges of each electrode. Where the offset value
                should correspond with (max_range_value + min_range_value) / 2. If no offset
                is given it will be automatically calculated from the driver configurations for
                activation electrode ranges. If it wants to be manually set, the offset
                variable should also be included in the dictionary.
            - ramp_time: float
                Time that will be taken before sending each batch to go from zero to the first
                point of the batch and to zero from the last point of the batch.
            - batch_time:
                Time that the sampling of each batch will take.
            - number_batches: int
                Number of batches that will be sampled. A default value of 3880 is reccommended.

    Returns
    ----------
    dict
        Sampling configuration dictionary with additional keys as follows:
            - batch_points: int
                Number of data points in the signal of a single batch
            - ramp_points: int
                Number of data points in the waiting signal between each batch.
    """
    configs = config_dict["input_data"]
    
    assert (
        config_dict['driver']["instruments_setup"]
        ["average_io_point_difference"]), (
            "Surrogate Model generation only supports when averaging_io_point_difference is true.")
    configs['sampling_frequency'] = config_dict["driver"]["instruments_setup"][
        "activation_sampling_frequency"]
    
    assert 'input_frequency' in configs, "Input frequency bases for the generated wave should be specified"
    assert type(configs['input_frequency']) is list or type(configs['input_frequency']) is np.ndarray, "Input frequency for the generated wave should be a list containing irrational frequencies per activation electrode"
   
    configs['input_frequency_numpy'] = get_frequency(configs)

    assert 'amplitude' in configs
    configs['amplitude_numpy'] = np.array(configs['amplitude'])[:, np.newaxis]

    assert 'offset' in configs
    configs['offset_numpy'] = np.array(configs['offset'])[:, np.newaxis]

    assert 'batch_time' in configs
    assert 'ramp_time' in configs
    assert 'sampling_frequency' in configs
    assert type(configs['batch_time']) is int or type(configs['batch_time']) is float

    configs[
        'batch_points'] = configs['batch_time'] * configs['sampling_frequency']
    configs[
        'ramp_points'] = configs['ramp_time'] * configs['sampling_frequency']
    return configs


def get_frequency(configs: dict) -> np.array:
    """
    Generate input frequency for each electrode.

    Parameters
    ----------
    configs: dict
        Sampling configurations, the dictionary has the following keys:
            - input_frequency: list
                Base frequencies of the input waves that will be created. In order to optimise
                coverage, irrational numbers are recommended. The list should have the same
                length as the activation electrode number. The input frequency list will be
                square rooted. E.g., for 7 activation electrodes:
                input_frequency = [2, 3, 5, 7, 13, 17, 19]
    
    Returns
    ---------
    np.array
        List of input frequencies for each eletrode.
    """
    aux = np.array(configs['input_frequency'])[:, np.newaxis]
    # TODO: Check the optimal value for 0.001
    return np.sqrt(aux[:configs['activation_electrode_no']]) * (
        0.001 * configs['sampling_frequency'])  # configs['factor']


def generate_sawtooth_multiple(input_range,
                               n_points,
                               up_direction=True) -> np.array:
    """Generates a simple sawtooth for a single channel (electrode). It goes from zero to a certain
    point (v_low), from that point to another point (v_max), and from that last point to zero again.
    The direction can be inverted using up_direction=True so that the sawtooth goes from zero to 
    v_max, from v_max to v_min, and from v_min to zero.

    Parameters
    ----------
    input_range : linst
        Minimum and maximum voltages that the sawtooth will achieve.
    n_points : int
        Number of points that the sawtooth will have.
    up_direction : bool, optional
        Direction of the sawtooth. If true, the sawtooth will go first up and then down. 
        If False, the sawtooth will go first down and then up. By default False.

    Returns
    -------
    np.array
        An array containing the two pointed sawtooth in a single dimension.
    """
    n_points = n_points / 2

    if up_direction:
        Input1 = np.linspace(
            0, input_range[0],
            int((n_points * input_range[0]) /
                (input_range[0] - input_range[1])))
        Input2 = np.linspace(input_range[0], input_range[1], int(n_points))
        Input3 = np.linspace(
            input_range[1], 0,
            int((n_points * input_range[1]) /
                (input_range[1] - input_range[0])))
    else:
        Input1 = np.linspace(
            0, input_range[1],
            int((n_points * input_range[1]) /
                (input_range[1] - input_range[0])))
        Input2 = np.linspace(input_range[1], input_range[0], int(n_points))
        Input3 = np.linspace(
            input_range[0], 0,
            int((n_points * input_range[0]) /
                (input_range[0] - input_range[1])))
    result = np.concatenate((Input1, Input2, Input3))
    if not (result.shape[0] == int(n_points * 2)):
        result = np.concatenate((result, np.array([0])))
    return result


def generate_sawtooth_simple(v_low: float,
                             v_high: float,
                             point_no: int,
                             up_direction: bool = False) -> np.array:
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
    assert v_low < v_high
    assert point_no % 2 == 0, 'Only an even point number is accepted.'
    point_no = int(point_no / 2)

    if up_direction:
        v_low, v_high = v_high, v_low

    ramp1 = np.linspace(0, v_low, round((point_no * v_low) / (v_low - v_high)))
    ramp2 = np.linspace(v_low, v_high, point_no)
    ramp3 = np.linspace(v_high, 0, round(
        (point_no * v_high) / (v_high - v_low)))

    result = np.concatenate((ramp1, ramp2, ramp3))
    return result


def generate_sinewave(n: int,
                      fs: float,
                      amplitude: float,
                      phase: float = 0) -> np.array:
    """
    Generates a sine wave that can be used for the input data.

    Parameters
    ----------
    n: int
        The number of points to generate in the sine wave.
    fs: float
        Frequency of the sine wave.
    amplitude: float
        Amplitude of the sine wave.
    phase: Optional[float]
        Phase offset of sinewave at t=0.
    
    Returns
    ---------
    np.array
        Generated sine wave.
    """
    freq = fs / n
    points = np.linspace(0, 1 / freq, n)
    phases = points * 2 * np.pi * freq

    return np.sin(phases + phase) * amplitude

def get_random_phase(activation_electrode_no=7):
    """
    Generates a list containing different random phases for each activation electrodes.
    It can be used before gathering the data, in order to randomize the phase of the
    input during the data acquisition after one or few sampling batches.

    Parameters
    ----------
    activation_electrode_no: int
        The number of activation electrodes for which the phase will be generated.

    Returns
    ---------
    list
        Generated phases for activation electrodes.
    """
    phase = (np.random.rand((activation_electrode_no)) -
             0.5) * 720  # Get values between -360 and 360 (degrees)
    phase *= (np.pi / 180)  # convert to radians
    return phase[:, np.newaxis]  #.tolist()

