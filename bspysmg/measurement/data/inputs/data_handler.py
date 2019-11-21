
import numpy as np
import math
from bspyproc.utils.pytorch import TorchUtils


def get_training_data(configs):
    '''
    This function loads the data and returns it in a format suitable for the NN to handle.
    Partitions the data into training, validation and test sets (if test_size is not None)
    Arguments data_dir and file_name required are strings: a path to the directory containing the data and the name of the data file respectively.
    Data structure loaded must be a .npz file with a directory having keys: 'inputs','outputs'.
    The inputs follow the convention that the first dimension is over CV configs and the second index is
    over input dimension, i.e. number of electrodes.
    '''
    path = configs['training_data_path']
    validation_size = configs['validation_size']
    steps = configs['steps']

    inputs, outputs, info_dictionary = load_data(path, steps)
    assert len(outputs) == len(inputs), 'Inputs and Outpus have NOT the same length'
    nr_samples = len(outputs)
    pc = nr_samples / info_dictionary['nr_raw_samples']
    print(f'Nr. of samples is {nr_samples}; {math.ceil(pc*100)}% of raw data')
    # Shuffle data
    shuffler = np.random.permutation(len(outputs))
    inputs = inputs[shuffler]
    outputs = outputs[shuffler]

    # Partition into training and validation sets ###
    n_val = int(nr_samples * validation_size)
    print('Size of VALIDATION set is ', n_val)
    print('Size of TRAINING set is ', nr_samples - n_val)

    # Training set
    inputs_train = inputs[n_val:]
    outputs_train = outputs[n_val:]

    # Sanity Check
    if not outputs_train.shape[0] == inputs_train.shape[0]:
        raise ValueError('Input and Output Batch Sizes do not match!')

    # Validation set
    if n_val > 0:
        inputs_val = inputs[:n_val]
        outputs_val = outputs[:n_val]
    else:
        inputs_val = None
        outputs_val = None

    return inputs_train, outputs_train, inputs_val, outputs_val, info_dictionary


def load_data(path, steps):
    print('Data loading from: \n' + path)
    with np.load(path, allow_pickle=True) as data:  # why was allow_pickle not required before? Do we need this?
        # TODO: change in data generation the key meta to info_dictionary
        info_dictionary = data['meta'].tolist()
        print(f'Metadata :\n {info_dictionary.keys()}')
        # Create from numpy arrays torch.tensors and send them to device
        inputs = TorchUtils.get_tensor_from_numpy(data['inputs'][::steps])  # shape: Nx#electrodes
        outputs = TorchUtils.get_tensor_from_numpy(data['outputs'][::steps])  # Outputs need dim Nx1
        print(f'Shape of outputs: {outputs.shape}; shape of inputs: {inputs.shape}')

    return inputs, outputs, info_dictionary
