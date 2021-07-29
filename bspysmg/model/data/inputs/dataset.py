import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from brainspy.utils.pytorch import TorchUtils


class ModelDataset(Dataset):
    def __init__(self, data_path, steps):
        self.inputs, targets, self.sampling_configs = self.load_data(data_path, steps)
        self.targets = (
            targets / self.sampling_configs["driver"]["amplification"]
        )
        self.inputs = TorchUtils.format(self.inputs)
        self.targets = TorchUtils.format(self.targets)

        assert len(self.inputs) == len(
            self.targets
        ), "Inputs and Outpus have NOT the same length"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (self.inputs[index, :], self.targets[index, :])

    def load_data(self, data_path, steps):
        print("\n* Loading data from file:\n" + data_path)
        with np.load(data_path, allow_pickle=True) as data:
            sampling_configs = data["info"].tolist()
            inputs = data["inputs"][::steps]
            outputs = data["outputs"][::steps]
            print(f"\t- Shape of inputs:  {inputs.shape}\n\t- Shape of outputs: {outputs.shape}\n")
            print(f"* Sampling configs has the following keys:\n\t{sampling_configs.keys()}\n")
        return inputs, outputs, sampling_configs

def get_info_dict(training_configs, sampling_configs):
    info_dict = {}
    info_dict["model_structure"] = training_configs["model_structure"].copy()
    info_dict["electrode_info"] = sampling_configs["electrode_info"].copy()
    del training_configs["model_structure"]
    info_dict["training_configs"] = training_configs.copy()
    del sampling_configs["electrode_info"]
    info_dict["sampling_configs"] = sampling_configs.copy()
    return info_dict


def load_data(configs):
    # Load dataset
    # Only training configs will be taken into account for info dict
    # For ranges and etc.
    datasets = []
    info_dict = None
    amplification = None
    dataset_names = ['train','validation','test']
    for i in range(len(configs['data']['dataset_paths'])):
        dataset = ModelDataset(configs['data']['dataset_paths'][i], configs['data']['steps'])
        
        if i > 0:
            amplification_aux = TorchUtils.format(info_dict["sampling_configs"]["driver"][
                "amplification"
            ])
            assert (torch.eq(amplification_aux, amplification).all(), 
            "Amplification correction factor should be the same for all datasets. Check if all datasets come from the same setup.")
            info_dict[dataset_names[i]+'_sampling_configs'] = dataset.sampling_configs
        else:
            info_dict = get_info_dict(configs, dataset.sampling_configs)
        amplification = TorchUtils.format(info_dict["sampling_configs"]["driver"][
                "amplification"
            ])
        datasets.append(dataset)

    # Create dataloaders
    dataloaders = []
    shuffle = [True, False, False]
    for i in range(len(datasets)):
        if len(datasets[i]) != 0:
            dataloaders.append(
                DataLoader(
                    dataset=datasets[i],
                    batch_size=configs["data"]["batch_size"],
                    num_workers=configs["data"]["worker_no"],
                    pin_memory=configs["data"]["pin_memory"],
                    shuffle=shuffle[i],
                )
            )
        else:
            dataloaders.append(None)
    return dataloaders, amplification, info_dict
