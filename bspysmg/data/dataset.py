import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from brainspy.utils.pytorch import TorchUtils
from typing import Tuple, List


class ModelDataset(Dataset):
    def __init__(self,
                 filename: str,
                 steps: int = 1,
                 tag: str = 'train') -> None:
        """Initialisation of the dataset. It loads a posprocessed_data.npz file into memory.
        The targets of this file are divided by the amplification correction factor, so that
        data is made setup independent.

        Parameters
        ----------
        filename : str
            Folder and filename where the posprocessed_data.npz is.
        steps : int
            It allows to skip parts of the data when loading it into memory. The number indicates
            how many items will be skipped in between. By default, step number is one (no values
            are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
            inputs taken into account would be: [0, 2, 4, 6].
        tag : str
            Name of the dataset. I.e., train, validation or test.

        Storage
        ----------
            The postprocessed data is a .npz file called postprocessed_data.npz
            with keys: inputs, outputs and info (dict)
                - The input(s) is(are) gathered for all activation electrodes. The units is in
                    Volts.
                - The output(s) is(are) gathered from all the readout electrodes. The units are in
                    nA.
                The output data is raw. Additional amplification correction might be needed, this is
                left for the user to decide.

            Data structure of output and input are arrays of NxD, where N is the number of samples
            and D is the dimension.

            The configs dictionary contains a copy of the configurations used for sampling the data.
            In addition, the configs dictionary has a key named electrode_info, which is created
            during the postprocessing step. The electrode_info key contains the following keys:
                * electrode_no: int
                    Total number of electrodes in the device
                * activation_electrodes: dict
                    - electrode_no: int
                        Number of activation electrodes used for gathering the data
                    - voltage_ranges: list
                        Voltage ranges used for gathering the data. It contains the ranges per
                        electrode, where the shape is (electrode_no,2). Being 2 the minimum and
                        maximum of the ranges, respectively.
                * output_electrodes: dict
                    - electrode_no : int
                        Number of output electrodes used for gathering the data
                    - clipping_value: list[float,float]
                        Value used to apply a clipping to the sampling data within the specified
                        values.
                    - amplification: float
                        Amplification correction factor used in the device to correct the
                        amplification applied to the output current in order to convert it into
                        voltage before its readout.

        """
        self.inputs, targets, self.sampling_configs = self.load_data_from_npz(
            filename, steps)
        self.targets = (targets /
                        self.sampling_configs["driver"]["amplification"])
        self.inputs = TorchUtils.format(self.inputs)
        self.targets = TorchUtils.format(self.targets)
        if tag not in ["train", "validation", "test"]:
            raise ValueError("tag should be one of [train, validation, test]")
        self.tag = tag

        assert len(self.inputs) == len(
            self.targets), "Inputs and Outpus have NOT the same length"

    def __len__(self) -> int:
        """Overwrittes the __len__ method from the super class torch.utils.data.

        Returns
        -------
        int
            Size of the whole dataset.
        """
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[np.array]:
        """Overwrittes the __getitem__ method from the super class torch.utils.data.
        The method supports fetching a data sample for a given key.

        Parameters
        ----------
        index : int
            Index corresponding to the place of the data in the dataset.

        Returns
        -------
        tuple
            Inputs and targets of the dataset corresponding to the given index.
        """
        return (self.inputs[index, :], self.targets[index, :])

    def load_data_from_npz(self, filename: str,
                           steps: int) -> Tuple[np.array, np.array, dict]:
        """
        Loads the inputs, targets and sampling configurations from a given postprocessed_data.npz
        file.

        Parameters
        ----------
        filename : str
            Folder and filename where the posprocessed_data.npz is.
        steps : int
            It allows to skip parts of the data when loading it into memory. The number indicates
            how many items will be skipped in between. By default, step number is one (no values
            are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
            inputs taken into account would be: [0, 2, 4, 6].

        Returns
        -------
        inputs : np.array
            Input waves sent to the activation electrodes of the device during sampling.
        outputs : np.array
            Raw output data from the readout electrodes of the device during sampling,
            corresponding to the input.
        sampling_configs : dict
            Dictionary containing the sampling configurations with which the data was
            acquired.

        Storage
        ----------
            The postprocessed data is a .npz file called postprocessed_data.npz
            with keys: inputs, outputs and info (dict)
                - The input(s) is(are) gathered for all activation electrodes. The units is in
                    Volts.
                - The output(s) is(are) gathered from all the readout electrodes. The units are in
                    nA.
                The output data is raw. Additional amplification correction might be needed, this is
                left for the user to decide.

            Data structure of output and input are arrays of NxD, where N is the number of samples
            and D is the dimension.

            The configs dictionary contains a copy of the configurations used for sampling the data.
            In addition, the configs dictionary has a key named electrode_info, which is created
            during the postprocessing step. The electrode_info key contains the following keys:
                * electrode_no: int
                    Total number of electrodes in the device
                * activation_electrodes: dict
                    - electrode_no: int
                        Number of activation electrodes used for gathering the data
                    - voltage_ranges: list
                        Voltage ranges used for gathering the data. It contains the ranges per
                        electrode, where the shape is (electrode_no,2). Being 2 the minimum and
                        maximum of the ranges, respectively.
                * output_electrodes: dict
                    - electrode_no : int
                        Number of output electrodes used for gathering the data
                    - clipping_value: list[float,float]
                        Value used to apply a clipping to the sampling data within the specified
                        values.
                    - amplification: float
                        Amplification correction factor used in the device to correct the
                        amplification applied to the output current in order to convert it into
                        voltage before its readout.
        """
        print("\n* Loading data from file:\n" + filename)
        # Pickle = True, since it also contains a dictionary.
        with np.load(filename, allow_pickle=True) as data:
            sampling_configs = dict(data["sampling_configs"].tolist())
            inputs = data["inputs"][::steps]
            outputs = data["outputs"][::steps]
            print(
                f"\t- Shape of inputs:  {inputs.shape}\n\t- Shape of outputs: {outputs.shape}\n"
            )
            print(
                f"* Sampling configs has the following keys:\n\t{sampling_configs.keys()}\n"
            )
        return inputs, outputs, sampling_configs


def get_info_dict(training_configs: dict, sampling_configs: dict) -> dict:
    """
    Retrieve the info dictionary given the training configs and the sampling configs.
    Note that the electrode_info key should be present in the sampling configs. This
    key is automatically generated when postprocessing the data.

    Parameters
    ----------
    training_configs : dict
        A copy of the configurations used for training the surrogate model.
    sampling_configs : dict
        A copy of the configurations used for sampling the training data.

    Returns
    -------
    info_dict
        This dictionary is required in order to initialise a surrogate
        model. It contains the following keys:
        * model_structure: dict
            The definition of the internal structure of the surrogate model, which is typically five
            fully-connected layers of 90 nodes each.
            - hidden_sizes : list
                A list containing the number of nodes of each layer of the surrogate model.
                E.g., [90,90,90,90,90]
            - D_in: int
                Number of input features of the surrogate model structure. It should correspond to
                the activation electrode number.
            - D_out: int
                Number of output features of the surrogate model structure. It should correspond to
                the readout electrode number.
        * electrode_info: dict
            It contains all the information required for the surrogate model about the electrodes.
                * electrode_no: int
                    Total number of electrodes in the device
                * activation_electrodes: dict
                    - electrode_no: int
                        Number of activation electrodes used for gathering the data
                    - voltage_ranges: list
                        Voltage ranges used for gathering the data. It contains the ranges per
                        electrode, where the shape is (electrode_no,2). Being 2 the minimum and
                        maximum of the ranges, respectively.
                * output_electrodes: dict
                    - electrode_no : int
                        Number of output electrodes used for gathering the data
                    - clipping_value: list[float,float]
                        Value used to apply a clipping to the sampling data within the specified
                        values.
                    - amplification: float
                        Amplification correction factor used in the device to correct the
                        amplification applied to the output current in order to convert it into
                        voltage before its readout.
        * training_configs: dict
            A copy of the configurations used for training the surrogate model.
        * sampling_configs : dict
            A copy of the configurations used for gathering the training data.
    """
    info_dict = {}
    info_dict["model_structure"] = training_configs["model_structure"].copy()
    info_dict["electrode_info"] = sampling_configs["electrode_info"].copy()
    del training_configs["model_structure"]
    info_dict["training_configs"] = training_configs.copy()
    del sampling_configs["electrode_info"]
    info_dict["sampling_configs"] = sampling_configs.copy()
    return info_dict


def get_dataloaders(
        configs: dict
) -> Tuple[List[torch.utils.data.DataLoader], float, dict]:
    """
    Loads all the datasets specified in the dataset_paths list key of the configurations dictionary
    and creates a dataloader.

    Parameters
    ----------
    configs : dict
        Surrogate model generation configurations.

        * results_base_dir: str
            Directory where the trained model and corresponding performance plots will be stored.
        * seed: int
            Sets the seed for generating random numbers to a non-deterministic random number.
        * hyperparameters:
            epochs: int
            learning_rate: float
        * model_structure: dict
            The definition of the internal structure of the surrogate model, which is typically five
            fully-connected layers of 90 nodes each.
            - hidden_sizes : list
                A list containing the number of nodes of each layer of the surrogate model.
                E.g., [90,90,90,90,90]
            - D_in: int
                Number of input features of the surrogate model structure. It should correspond to
                the activation electrode number.
            - D_out: int
                Number of output features of the surrogate model structure. It should correspond to
                the readout electrode number.
        * data:
        dataset_paths: list[str]
            A list of paths to the Training, Validation and Test datasets, stored as
            postprocessed_data.npz
        steps : int
            It allows to skip parts of the data when loading it into memory. The number indicates
            how many items will be skipped in between. By default, step number is one (no values
            are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
            inputs taken into account would be: [0, 2, 4, 6].
        batch_size: int
            How many samples will contain each forward pass.
        worker_no: int
            How many subprocesses to use for data loading. 0 means that the data will be loaded in
            the main process. (default: 0)
        pin_memory: boolean
            If True, the data loader will copy Tensors into CUDA pinned memory before returning
            them. If your data elements are a custom type, or your collate_fn returns a batch that
            is a custom type.
    Returns
    -------
    dataloaders : list[torch.utils.dataDataLoader]
        A list containing the corresponding training, validation and test datasets.

    """
    # Load dataset
    # Only training configs will be taken into account for info dict
    # For ranges and etc.
    assert configs['data']['dataset_paths'] != [], "Empty paths for datasets"
    assert isinstance(configs['data']['dataset_paths'], list), "Paths for datasets should be passed as a list"
    datasets = []
    info_dict = None
    amplification = None
    dataset_names = ['train', 'validation', 'test']
    for i in range(len(configs['data']['dataset_paths'])):
        dataset = ModelDataset(configs['data']['dataset_paths'][i],
                               steps=configs['data']['steps'],
                               tag=dataset_names[i])

        if i > 0:
            amplification_aux = TorchUtils.format(info_dict["sampling_configs"]["driver"]["amplification"])
            assert torch.eq(amplification_aux, amplification).all(), (
                "Amplification correction factor should be the same for all datasets."
                + "Check if all datasets come from the same setup.")
            info_dict[dataset_names[i] +
                      '_sampling_configs'] = dataset.sampling_configs
        else:
            info_dict = get_info_dict(configs, dataset.sampling_configs)
        amplification = TorchUtils.format(info_dict["sampling_configs"]["driver"]["amplification"])
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
                ))
        else:
            dataloaders.append(None)
    return dataloaders, amplification, info_dict
