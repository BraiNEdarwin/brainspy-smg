
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from brainspy.utils.pytorch import TorchUtils


class ModelDataset(Dataset):

    def __init__(self, configs):
        self.inputs, targets, self.info_dict = self.load_data(configs["data"])
        self.targets = targets / self.info_dict['processor']['driver']['amplification']
        self.inputs = TorchUtils.get_tensor_from_numpy(self.inputs).cpu()
        self.targets = TorchUtils.get_tensor_from_numpy(self.targets).cpu()
        assert len(self.inputs) == len(self.targets), 'Inputs and Outpus have NOT the same length'

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (self.inputs[index, :], self.targets[index, :])

    def load_data(self, configs):
        print('Data loading from: \n' + configs['postprocessed_data_path'])
        with np.load(configs['postprocessed_data_path'], allow_pickle=True) as data:  # why was allow_pickle not required before? Do we need this?
            # TODO: change in data generation the key meta to info_dictionary
            info_dictionary = data['info'].tolist()
            print(f'Metadata :\n {info_dictionary.keys()}')
            # Create from numpy arrays torch.tensors and send them to device
            inputs = data['inputs'][::configs['steps']]  # TorchUtils.get_tensor_from_numpy(data['inputs'][::configs['steps']])  # shape: Nx#electrodes
            outputs = data['outputs'][::configs['steps']]  # TorchUtils.get_tensor_from_numpy(data['outputs'][::configs['steps']])  # Outputs need dim Nx1
            print(f'Shape of outputs: {outputs.shape}; shape of inputs: {inputs.shape}')

        return inputs, outputs, info_dictionary


def load_data(configs):
    # Load dataset
    dataset = ModelDataset(configs)

    amplification = dataset.info_dict['processor']['driver']['amplification']

    # Split dataset
    split_percentages = [int(len(dataset) * configs['hyperparameters']['split_percentages'][0]), int(len(dataset) * configs['hyperparameters']['split_percentages'][1]), int(len(dataset) * configs['hyperparameters']['split_percentages'][2])]
    while len(dataset) != sum(split_percentages):
        split_percentages[0] += 1

    datasets = random_split(dataset, split_percentages)

    # Create dataloaders
    # If length of the dataset is not divisible by the batch_size, it will drop the last batch.
    dataloaders = [DataLoader(dataset=datasets[i], batch_size=configs['hyperparameters']['batch_size'], num_workers=configs['hyperparameters']['worker_no'], pin_memory=configs['hyperparameters']['pin_memory'], drop_last=True, shuffle=True) for i in range(len(datasets))]

    return dataloaders, amplification, dataset.info_dict
