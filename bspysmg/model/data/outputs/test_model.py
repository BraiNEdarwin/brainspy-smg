import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from bspyalgo.utils.io import save, create_directory
from bspysmg.model.data.plots.model_results_plotter import plot_all
from bspyproc.utils.pytorch import TorchUtils


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


def get_previous_path(test_data_path):
    path = test_data_path.split('/')
    return path[-2]


def get_error(model, model_data_path, test_data_path, steps=1, batch_size=2700000, model_name='trained_network_with_mse'):
    with torch.no_grad():
        inputs, targets, info = load_data(test_data_path, steps)
        amplification = np.array(info['processor']['amplification'])
        error = np.zeros_like(targets)
        prediction = np.zeros_like(targets)
        #inputs = TorchUtils.get_tensor_from_numpy(inputs)
        # targets = TorchUtils.get_tensor_from_numpy(targets)

    i_start = 0
    i_end = batch_size
    threshold = (inputs.shape[0] - batch_size)
    while i_end < inputs.shape[0]:
        prediction[i_start:i_end] = TorchUtils.get_numpy_from_tensor(model(TorchUtils.get_tensor_from_numpy(inputs[i_start:i_end]))) * amplification
        error[i_start:i_end] = prediction[i_start:i_end] - targets[i_start:i_end]
        i_start += batch_size
        i_end += batch_size
        if i_end > threshold and i_end < inputs.shape[0]:
            i_end = inputs.shape[0]

    path = create_directory(os.path.join(get_main_path(model_data_path), get_previous_path(test_data_path)))
    mse = plot_all(targets, prediction, path, name='TEST')
    # save('numpy', os.path.join(path, 'error.npz'), error=TorchUtils.get_numpy_from_tensor(error), prediction=TorchUtils.get_numpy_from_tensor(prediction), targets=TorchUtils.get_numpy_from_tensor(targets), test_mse=TorchUtils.get_numpy_from_tensor(mse))

    return mse
