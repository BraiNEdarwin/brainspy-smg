from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.measurement.data.inputs.data_handler import load_data
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.simulation.network import TorchModel
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def get_error(model_path, test_data_path, steps=1):

    with torch.no_grad():
        model = TorchModel(model_path)
        INPUTS_TEST, TARGETS_TEST, INFO_DICT = load_data(test_data_path, steps)
    prediction = model.get_output(INPUTS_TEST)
    TARGETS_TEST = TorchUtils.get_numpy_from_tensor(TARGETS_TEST)
    error = (prediction - TARGETS_TEST)
    MSE = np.mean(error**2)
    print(f'MSE on Test Set of trained NN model: {MSE}')
    dir_path, model_name = os.path.split(model_path)
    model_name = os.path.splitext(model_name)[0]

    # PLOT PREDICTED VS TRUE VALUES and ERROR HIST
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(TARGETS_TEST, prediction, '.')
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    targets_and_prediction_array = np.concatenate((TARGETS_TEST, prediction))
    min_out = np.min(targets_and_prediction_array)
    max_out = np.max(targets_and_prediction_array)
    plt.plot(np.linspace(min_out, max_out), np.linspace(min_out, max_out), 'k')
    plt.title('Predicted vs True values')
    plt.subplot(1, 2, 2)
    plt.hist(np.reshape(error, error.size), 500)
    plt.title('Scaled error histogram')
    fig_loc = os.path.join(dir_path, f'Test_Error_{model_name}')
    plt.savefig(fig_loc)
    print(f'Test Error Figures saved in {dir_path}')
    # PLOT TEST ERROR VS OUTPUT
    plt.figure()
    plt.plot(TARGETS_TEST, error, '.')
    plt.plot(np.linspace(TARGETS_TEST.min(), TARGETS_TEST.max(), len(error)), np.zeros_like(error))
    plt.title('Error vs Output')
    plt.xlabel('Output')
    plt.ylabel('Error')
    fig_loc = os.path.join(dir_path, f'TestError_vs_Output_{model_name}')
    plt.savefig(fig_loc)
    print(f'Test Error Figures saved in {dir_path}')

    return MSE


def get_control_voltages(path_to_configs):
    print("Training DNPU for boolean logic")
    predictor = get_algorithm(path_to_configs)
    # TODO: implement tasks
    raise NotImplementedError("Tasks are not implemeted!")
