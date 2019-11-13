from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.measurement.data.inputs import get_nn_data


def get_error(model_generator):
    INPUTS_TEST, TARGETS_TEST = get_nn_data(model_generator.configs)
    NotImplementedError, 'Test error is not implemented'

    # calculate error ...


def get_control_voltages(configs):
    predictor = get_algorithm(configs)
    NotImplementedError, 'Predicting CV is not implemented'
