from bspyalgo.algorithm_manager import get_algorithm
# from bspysmg.model.data.inputs.data_handler import load_data
from bspyproc.processors.simulation.network import TorchModel
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from bspyalgo.utils.io import save, create_directory
from bspysmg.model.data.plots.model_results_plotter import plot_all


def load_data(path, steps):
    print('Data loading from: \n' + path)
    with np.load(path, allow_pickle=True) as data:  # why was allow_pickle not required before? Do we need this?
        # TODO: change in data generation the key meta to info_dictionary
        info_dictionary = data['info'].tolist()
        print(f'Metadata :\n {info_dictionary.keys()}')
        inputs = data['inputs'][::steps]  # shape: Nx#electrodes
        outputs = data['outputs'][::steps]  # Outputs need dim Nx1
        print(f'Shape of outputs: {outputs.shape}; shape of inputs: {inputs.shape}')

    return inputs, outputs, info_dictionary


def get_main_path(model_path):
    path = model_path.split('/')
    del path[len(path) - 1]
    if 'reproducibility' in path:
        del path[len(path) - 1]
    return os.path.join(*path)


def get_error(model_path, test_data_path, steps=1, batch_size=2700000, model_name='trained_network_with_mse'):
    results_dir = 'final_model'
    with torch.no_grad():
        model = TorchModel({'torch_model_dict': model_path})
        INPUTS_TEST, TARGETS_TEST, INFO_DICT = load_data(test_data_path, steps)
        error = np.zeros_like(TARGETS_TEST)
        prediction = np.zeros_like(TARGETS_TEST)
    i_start = 0
    i_end = batch_size
    threshold = (INPUTS_TEST.shape[0] - batch_size)
    while i_end < INPUTS_TEST.shape[0]:
        prediction[i_start:i_end] = model.get_output(INPUTS_TEST[i_start:i_end])
        error[i_start:i_end] = prediction[i_start:i_end] - TARGETS_TEST[i_start:i_end]
        i_start += batch_size
        i_end += batch_size
        if i_end > threshold and i_end < INPUTS_TEST.shape[0]:
            i_end = INPUTS_TEST.shape[0]

    mse = np.mean(error**2)
    print(f'MSE on Test Set of trained NN model: {mse}')
    # dir_path, model_name = os.path.split(model_path)
    # model_name = os.path.splitext(model_name)[0]
    # model.info['data_info']['mse'] = mse.item()

    path = create_directory(os.path.join(get_main_path(model_path), results_dir))
    save('numpy', os.path.join(path, 'error.npz'), error=error, prediction=prediction, targets=TARGETS_TEST, test_mse=mse)
    # save('torch', os.path.join(path, model_name + '.pt'), timestamp=False, data=model)
    plot_all(TARGETS_TEST, prediction, path, name='TEST')
