
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from brainspy.utils.pytorch import TorchUtils


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


def load_data(configs):
    # Load dataset
    dataset = ModelDataset(configs)
    amplification = dataset.info_dict['processor']['amplification']

    # Split dataset
    split_percentages = [int(len(dataset) * configs['hyperparameters']['split_percentages'][0]), int(len(dataset) * configs['hyperparameters']['split_percentages'][1]), int(len(dataset) * configs['hyperparameters']['split_percentages'][2])]
    if len(dataset) != sum(split_percentages):
        split_percentages[0] += 1
    datasets = random_split(dataset, split_percentages)

    # Create dataloaders
    # If length of the dataset is not divisible by the batch_size, it will drop the last batch.
    dataloaders = [DataLoader(dataset=datasets[i], batch_size=configs['hyperparameters']['batch_size'], num_workers=configs['hyperparameters']['worker_no'], shuffle=True) for i in range(len(datasets))]

    return dataloaders, amplification
