from bspyalgo.algorithm_manager import get_algorithm
# from bspysmg.model.data.inputs.data_handler import load_data
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.simulation.network import TorchModel
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from bspyalgo.utils.io import save, create_directory


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
        error = np.zeros_like(INPUTS_TEST)
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

    MSE = np.mean(error**2)
    print(f'MSE on Test Set of trained NN model: {MSE}')
    dir_path, model_name = os.path.split(model_path)
    model_name = os.path.splitext(model_name)[0]

    model.info['data_info']['mse'] = MSE.item()

    path = create_directory(os.path.join(get_main_path(model_path), results_dir))
    save('numpy', os.path.join(path, 'error.npz'), error=error, prediction=prediction, targets=TARGETS_TEST)
    save('torch', os.path.join(path, model_name + '.pt'), timestamp=False, data=model)

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
    plt.title(f'Predicted vs True values:\n MSE {MSE}')
    plt.subplot(1, 2, 2)
    plt.hist(np.reshape(error, error.size), 500)
    x_lim = 0.25 * np.max([np.abs(error.min()), error.max()])
    plt.xlim([-x_lim, x_lim])
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


if __name__ == '__main__':
    error = get_error('/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/training_1080batches_50s_50Hz_2019_12_05_172238/model_1200ep_10-4lr_mb128_2019_12_10_170729/trained_network.pt',
                      '/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/BRAINSPY_DATA/training_1080batches_50s_50Hz_NonZeroPhase_2019_12_10_164417/training_data.npz')
