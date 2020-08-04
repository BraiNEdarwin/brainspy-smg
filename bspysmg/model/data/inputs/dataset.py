
import os
import numpy as np
from torch.utils.data import Dataset
from bspyproc.utils.pytorch import TorchUtils


class ModelDataset(Dataset):

    def __init__(self, configs):
        path = os.path.join(configs["results_base_dir"], configs["data"]['training_data_path'])
        self.inputs, targets, self.info_dict = self.load_data(path, configs["data"]['steps'])
        self.targets = targets / self.info_dict['processor']['amplification']
        assert len(self.inputs) == len(self.targets), 'Inputs and Outpus have NOT the same length'

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (self.inputs[index, :], self.targets[index, :])

    def load_data(self, path, steps):
        print('Data loading from: \n' + path)
        with np.load(path, allow_pickle=True) as data:  # why was allow_pickle not required before? Do we need this?
            # TODO: change in data generation the key meta to info_dictionary
            info_dictionary = data['info'].tolist()
            print(f'Metadata :\n {info_dictionary.keys()}')
            # Create from numpy arrays torch.tensors and send them to device
            inputs = TorchUtils.get_tensor_from_numpy(data['inputs'][::steps])  # shape: Nx#electrodes
            outputs = TorchUtils.get_tensor_from_numpy(data['outputs'][::steps])  # Outputs need dim Nx1
            print(f'Shape of outputs: {outputs.shape}; shape of inputs: {inputs.shape}')

        return inputs, outputs, info_dictionary
