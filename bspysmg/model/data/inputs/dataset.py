import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from brainspy.utils.pytorch import TorchUtils


class ModelDataset(Dataset):
    def __init__(self, configs, train=True):
        self.train = train
        self.inputs, targets, self.sampling_configs = self.load_data(configs["data"])
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

    def load_data(self, configs):
        data_path = self.get_data_path(configs)
        print("\n* Loading data from file:\n" + data_path)
        with np.load(
            data_path, allow_pickle=True
        ) as data:  # why was allow_pickle not required before? Do we need this?
            # TODO: change in data generation the key meta to info_dictionary
            sampling_configs = data["info"].tolist()
            
            # Create from numpy arrays torch.tensors and send them to device
            inputs = data["inputs"][
                :: configs["steps"]
            ]  # TorchUtils.get_tensor_from_numpy(data['inputs'][::configs['steps']])  # shape: Nx#electrodes
            outputs = data["outputs"][
                :: configs["steps"]
            ]  # TorchUtils.get_tensor_from_numpy(data['outputs'][::configs['steps']])  # Outputs need dim Nx1
            print(f"\t- Shape of inputs:  {inputs.shape}\n\t- Shape of outputs: {outputs.shape}\n")
            print(f"* Sampling configs has the following keys:\n\t{sampling_configs.keys()}\n")
        return inputs, outputs, sampling_configs

    def get_data_path(self, configs):
        if self.train:
            return configs["train_data_path"]
        else:
            return configs['test_data_path']

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
    dataset = ModelDataset(configs)
    info_dict = get_info_dict(configs, dataset.sampling_configs)

    amplification = TorchUtils.format(info_dict["sampling_configs"]["driver"][
        "amplification"
    ])
    if configs["data"]["split_percentages"][0] < 1:
        len_val = int(len(dataset) * configs["data"]["split_percentages"][1])
    
        # Split dataset
        split_length = [
            len(dataset) - len_val,
            len_val
        ]

        datasets = random_split(dataset, split_length)
    else:
        datasets = []
        datasets.append(dataset)
        datasets.append([])

    if 'test_data_path' in configs['data']:
        test_dataset = ModelDataset(configs, train=False)
        datasets.append(test_dataset)

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
