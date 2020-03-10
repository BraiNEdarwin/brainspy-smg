import numpy as np
import torch
import os
from bspyalgo.utils.io import create_directory
from bspysmg.model.data.plots.model_results_plotter import plot_all
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.simulation.surrogate import SurrogateModel


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


def get_error(model_data_path, test_data_path, steps=1, batch_size=2048):

    inputs, targets, info = load_data(test_data_path, steps)
    error = np.zeros_like(targets)
    prediction = np.zeros_like(targets)
    model = SurrogateModel({'torch_model_dict': model_data_path})
    with torch.no_grad():
        i_start = 0
        i_end = batch_size
        threshold = (inputs.shape[0] - batch_size)
        while i_end <= inputs.shape[0]:
            prediction[i_start:i_end] = TorchUtils.get_numpy_from_tensor(model(TorchUtils.get_tensor_from_numpy(inputs[i_start:i_end])))
            error[i_start:i_end] = prediction[i_start:i_end] - targets[i_start:i_end]
            i_start += batch_size
            i_end += batch_size
            if i_end > threshold and i_end < inputs.shape[0]:
                i_end = inputs.shape[0]
        main_path = os.path.dirname(os.path.dirname(model_data_path))
        path = create_directory(os.path.join(main_path, 'test_model'))
        mse = plot_all(targets, prediction, path, name='TEST')

    return mse


if __name__ == "__main__":

    model_path = '/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/BRAINSPY_DATA/BRAINS/Brains_2020_03_06_182728/BRAINS_500ep_0.00005lr_mb128_2020_03_09_213229/reproducibility/model.pt'
    test_data_path = '/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/BRAINSPY_DATA/BRAINS/Brains_phase1_2020_03_09_094534/postprocessed_data.npz'
    mse = get_error(model_path, test_data_path)
    print(f'MSE: {mse}')
