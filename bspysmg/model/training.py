"""
File containing functions for training a surrogate model in pytorch taking into account the error in nano amperes.
"""
import os
import torch
import matplotlib.pyplot as plt

import numpy as np

# from brainspy.algorithm_manager import get_algorithm
# from bspysmg.model.data.inputs.data_handler import get_training_data
from tqdm import tqdm

from torch.optim import Adam
from torch.nn import MSELoss

from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.io import create_directory_timestamp
from brainspy.processors.simulation.model import NeuralNetworkModel
from bspysmg.data.dataset import get_dataloaders
from bspysmg.utils.plots import plot_error_vs_output, plot_error_hist
from typing import Tuple, List


def init_seed(configs: dict) -> None:
    """
    Initializes a random seed for training. A random seed is a starting point for pseudorandom
    number generator algorithms which is used for reproducibility.
    Also see - https://pytorch.org/docs/stable/notes/randomness.html  

    Parameters
    ----------
    configs : dict
        Training configurations with the following keys:

        1. seed:  int [Optional]
        The desired seed for the random number generator. If the dictionary does not contain 
        this key, a deterministic random seed will be applied, and added to the key 'seed' in 
        the dictionary.
    """
    if "seed" in configs:
        seed = configs["seed"]
    else:
        seed = None

    seed = TorchUtils.init_seed(seed, deterministic=True)
    configs["seed"] = seed


def generate_surrogate_model(
        configs: dict,
        custom_model: torch.nn.Module = NeuralNetworkModel,
        criterion: torch.nn.modules.loss._Loss = MSELoss(),
        custom_optimizer: torch.optim.Optimizer = Adam,
        main_folder: str = "training_data") -> None:
    """
    It loads the training and validation datasets from the npz file specified
    in the field data/dataset_paths of the configs dictionary. These npz files
    can be created when running the postprocessing of the sampling data. The
    method will train a neural network with the structure specified in the
    model_structure field of the configs using the loaded training and validation
    datasets. It provides, on the specified saving directory, a trained model,
    the plots of the training performance, and the error of the model.

    Note that the method will only save a model (in each epoch) if current validation
    loss is less than the validation loss from the previous epoch.

    Parameters
    ----------
    configs : dict
        Training configurations for training a model with following keys:
        
        1. results_base_dir: str
        Directory where the trained model and corresponding performance plots will be stored.

        2. seed: int
        Sets the seed for generating random numbers to a non-deterministic random number.

        3. hyperparameters:
        epochs: int
        learning_rate: float
        
        4. model_structure: dict
        The definition of the internal structure of the surrogate model, which is typically five
        fully-connected layers of 90 nodes each.

        4.1 hidden_sizes : list
        A list containing the number of nodes of each layer of the surrogate model.
        E.g., [90,90,90,90,90]

        4.2 D_in: int
        Number of input features of the surrogate model structure. It should correspond to
        the activation electrode number.

        4.3 D_out: int
        Number of output features of the surrogate model structure. It should correspond to
        the readout electrode number.

        5. data:
        5.1 dataset_paths: list[str]
        A list of paths to the Training, Validation and Test datasets, stored as
        postprocessed_data.npz

        5.2 steps : int
        It allows to skip parts of the data when loading it into memory. The number indicates
        how many items will be skipped in between. By default, step number is one (no values
        are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
        inputs taken into account would be: [0, 2, 4, 6].

        5.3 batch_size: int
        How many samples will contain each forward pass.

        5.4 worker_no: int
        How many subprocesses to use for data loading. 0 means that the data will be loaded in
        the main process. (default: 0)

        5.5 pin_memory: boolean
        If True, the data loader will copy Tensors into CUDA pinned memory before returning
        them. If your data elements are a custom type, or your collate_fn returns a batch that
        is a custom type.
        custom_model : custom model of type torch.nn.Module
            Model to be trained.
        criterion : <method>
            Loss function that will be used to train the model.
        custom_optimizer : torch.optim.Optimizer
            Optimization method used to train the model which decreases model's loss.
        save_dir : string [Optional]
            Name of the path where the trained model is to be saved.
    
    Return
    ------
    saved_dir: str 
        Directory where the surrogate model was saved.
    """
    # Initialise seed and create data directories
    init_seed(configs)
    results_dir = create_directory_timestamp(configs["results_base_dir"],
                                             main_folder)

    # Get training, validation and test data
    # Get amplification of the device and the info
    dataloaders, amplification, info_dict = get_dataloaders(configs)

    # Initilialise model
    model = custom_model(info_dict["model_structure"])
    # model.set_info_dict(info_dict)
    model = TorchUtils.format(model)

    # Initialise optimiser
    optimizer = custom_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs["hyperparameters"]["learning_rate"],
        betas=(0.9, 0.75),
    )

    # Whole training loop
    model, performances, saved_dir = train_loop(
        model,
        info_dict,
        (dataloaders[0], dataloaders[1]),
        criterion,
        optimizer,
        configs["hyperparameters"]["epochs"],
        amplification,
        save_dir=results_dir,
    )

    # Plot results
    labels = ["TRAINING", "VALIDATION", "TEST"]
    for i in range(len(dataloaders)):
        if dataloaders[i] is not None:
            loss = postprocess(
                dataloaders[i],
                model,
                criterion,
                amplification,
                results_dir,
                label=labels[i],
            )

    plt.figure()
    plt.plot(TorchUtils.to_numpy(performances[0]))
    if len(performances) > 1 and not len(performances[1]) == 0:
        plt.plot(TorchUtils.to_numpy(performances[1]))
    if dataloaders[-1].tag == 'test':
        plt.plot(np.ones(len(performances[-1])) * TorchUtils.to_numpy(loss))
        plt.title("Training profile /n Test loss : %.6f (nA)" % loss)
    else:
        plt.title("Training profile")
    if not len(performances[1]) == 0:
        plt.legend(["training", "validation"])
    plt.xlabel("Epoch no.")
    plt.ylabel("RMSE (nA)")
    plt.savefig(os.path.join(results_dir, "training_profile"))
    if not dataloaders[-1].tag == 'train':
        training_data = torch.load(
            os.path.join(results_dir, "training_data.pt"))
        training_data['test_loss'] = loss
        torch.save(training_data, os.path.join(results_dir,
                                               "training_data.pt"))
    # print("Model saved in :" + results_dir)
    return saved_dir


def train_loop(
    model: torch.nn.Module,
    info_dict: dict,
    dataloaders: List[torch.utils.data.DataLoader],
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    amplification: float,
    start_epoch: int = 0,
    save_dir: str = None,
    early_stopping: bool = True,
) -> Tuple[torch.nn.Module, List[float]]:
    """
    Performs the training of a model and returns the trained model, training loss
    validation loss. It also saves the model in each epoch if current validation
    loss is less than the previous validation loss.

    Parameters
    ----------
    model : custom model of type torch.nn.Module
        Model to be trained.
    info_dict : dict
        The dictionary used for initialising the surrogate model. It has the following keys:
        1. model_structure: dict
        The definition of the internal structure of the surrogate model, which is typically five
        fully-connected layers of 90 nodes each.

        1.1 hidden_sizes : list
        A list containing the number of nodes of each layer of the surrogate model.
        E.g., [90,90,90,90,90]

        1.2 D_in: int
        Number of input features of the surrogate model structure. It should correspond to
        the activation electrode number.

        1.3 D_out: int
        Number of output features of the surrogate model structure. It should correspond to
        the readout electrode number.

        2. electrode_info: dict
        It contains all the information required for the surrogate model about the electrodes.
        2.1 electrode_no: int
        Total number of electrodes in the device

        2.2 activation_electrodes: dict
        2.2.1 electrode_no: int
        Number of activation electrodes used for gathering the data
        
        2.2.2 voltage_ranges: list
        Voltage ranges used for gathering the data. It contains the ranges per
        electrode, where the shape is (electrode_no,2). Being 2 the minimum and
        maximum of the ranges, respectively.

        2.3 output_electrodes: dict

        2.3.1 electrode_no : int
        Number of output electrodes used for gathering the data

        2.3.2 clipping_value: list[float,float]
        Value used to apply a clipping to the sampling data within the specified
        values.

        2.3.3 amplification: float
        Amplification correction factor used in the device to correct the
        amplification applied to the output current in order to convert it into
        voltage before its readout.

        3. training_configs: dict
        A copy of the configurations used for training the surrogate model.

        4. sampling_configs : dict
        A copy of the configurations used for gathering the training data.
    dataloaders :  list
        A list containing a single PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Loss function that will be used to train the model.
    optimizer : torch.optim.Optimizer
        Optimization method used to train the model which decreases model's loss.
    epochs : int
        The number of iterations for which the model is to be trained.
    amplification: float
        Amplification correction factor used in the device to correct the amplification
        applied to the output current in order to convert it into voltage before its
        readout.
    start_epoch : int [Optional]
        The starting value of the epochs.
    save_dir : string [Optional]
        Name of the path and file where the trained model is to be saved.
    early_stopping : bool [Optional]
        If this is set to true, early stopping algorithm is used during the training
        of the model.
        Also see - https://medium.com/analytics-vidhya/early-stopping-with-pytorch-to-
        restrain-your-model-from-overfitting-dce6de4081c5

    Returns
    -------
    model
        Trained model
    losses
        list of training loss and validation loss.
    saved_dir
        directory where the model was saved
    """
    if start_epoch > 0:
        start_epoch += 1

    train_losses, val_losses = TorchUtils.format([]), TorchUtils.format([])
    min_val_loss = np.inf

    for epoch in range(epochs):
        print("\nEpoch: " + str(epoch))
        model, running_loss = default_train_step(model, dataloaders[0],
                                                 criterion, optimizer)
        running_loss = running_loss**(1 / 2)
        running_loss *= amplification
        train_losses = torch.cat((train_losses, running_loss.unsqueeze(dim=0)),
                                 dim=0)
        description = "Training loss (RMSE): {:.6f} (nA)\n".format(
            train_losses[-1].item())

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            val_loss = default_val_step(model, dataloaders[1], criterion)
            val_loss = val_loss**(1 / 2)
            val_loss *= amplification
            val_losses = torch.cat((val_losses, val_loss.unsqueeze(dim=0)),
                                   dim=0)
            description += "Validation loss (RMSE): {:.6f} (nA)\n".format(
                val_losses[-1].item())
            # Save only when peak val performance is reached
            if (save_dir is not None and early_stopping
                    and val_losses[-1] < min_val_loss):
                min_val_loss = val_losses[-1]
                description += "Model saved in: " + save_dir
                # torch.save(model, os.path.join(save_dir, "model.pt"))
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "info": info_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "min_val_loss": min_val_loss,
                    },
                    os.path.join(save_dir, "training_data.pt"),
                )

        print(description)
        # looper.set_description(description)

    # TODO: Add a save instruction and a stopping criteria
    # if stopping_criteria(train_losses, val_losses):
    #     break
    print("\nFinished training model. ")
    print("Model saved in: " + save_dir)
    if (save_dir is not None and early_stopping and dataloaders[1] is not None
            and len(dataloaders[1]) > 0):
        training_data = torch.load(os.path.join(save_dir, "training_data.pt"))
        model.load_state_dict(training_data["model_state_dict"])
        print("Min validation loss (RMSE): {:.6f} (nA)\n".format(
            min_val_loss.item()))
    else:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "info": info_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "min_val_loss": min_val_loss,
            },
            os.path.join(save_dir, "training_data.pt"),
        )

    return model, [train_losses, val_losses], save_dir


def default_train_step(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer) -> Tuple[torch.nn.Module, float]:
    """
    Performs the training step of a model within a single epoch and returns the
    current loss and current trained model.

    Parameters
    ----------
    model : custom model of type torch.nn.Module
        Model to be trained.
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Loss function that will be used to train the model.
    optimizer : torch.optim.Optimizer
        Optimization method used to train the model which decreases model's loss.

    Returns
    -------
    tuple
        Trained model and training loss for the current epoch.
    """
    running_loss = 0
    model.train()
    loop = tqdm(dataloader)
    for inputs, targets in loop:
        inputs, targets = to_device(inputs), to_device(targets)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.shape[0]
        loop.set_postfix(batch_loss=loss.item())
    running_loss /= len(dataloader.dataset)
    return model, running_loss


def default_val_step(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     criterion: torch.nn.modules.loss._Loss) -> float:
    """
    Performs the validation step of a model within a single epoch and returns
    the validation loss.

    Parameters
    ----------
    model : custom model of type torch.nn.Module
        Model to be trained.
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Loss function that will be used to train the model.

    Returns
    -------
    float
        Validation loss for the current epoch.
    """
    with torch.no_grad():
        val_loss = 0
        model.eval()
        loop = tqdm(dataloader)
        for inputs, targets in loop:
            inputs, targets = to_device(inputs), to_device(targets)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            val_loss += loss.item() * inputs.shape[0]
            loop.set_postfix(batch_loss=loss.item())
        val_loss /= len(dataloader.dataset)
    return val_loss


def postprocess(dataloader: torch.utils.data.DataLoader,
                model: torch.nn.Module, criterion: torch.nn.modules.loss._Loss,
                amplification: float, results_dir: str, label: str) -> float:
    """
    Plots error vs output and error histogram for given dataset and saves it to
    specified directory.

    Parameters
    ----------
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    model : custom model of type torch.nn.Module
        Model to be trained.
    criterion : <method>
        Loss function that will be used to train the model.
    amplification: float
        Amplification correction factor used in the device to correct the amplification
        applied to the output current in order to convert it into voltage before its
        readout.
    results_dir : string
        Name of the path and file where the plots are to be saved.
    label : string
        Name of the dataset. I.e., train, validation or test.

    Returns
    -------
    float
        Mean Squared error evaluated on given dataset.
    """
    print(f"Postprocessing {label} data ... ")
    # i = 0
    running_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for inputs, targets in tqdm(dataloader):
            inputs, targets = to_device(inputs), to_device(targets)
            predictions = model(inputs)
            all_targets.append(amplification * targets)
            all_predictions.append(amplification * predictions)
            loss = criterion(predictions, targets)
            running_loss += loss * inputs.shape[0]  # sum up batch loss

    running_loss /= len(dataloader.dataset)
    running_loss = running_loss * (amplification**2)

    print(label.capitalize() +
          " loss (MSE): {:.6f} (nA)".format(running_loss.item()))
    print(
        label.capitalize() +
        " loss (RMSE): {:.6f} (nA)\n".format(torch.sqrt(running_loss).item()))

    all_targets = TorchUtils.to_numpy(torch.cat(all_targets, dim=0))
    all_predictions = TorchUtils.to_numpy(torch.cat(all_predictions, dim=0))

    error = all_targets - all_predictions

    plot_error_vs_output(
        all_targets,
        error,
        results_dir,
        name=label + "_error_vs_output",
    )
    plot_error_hist(
        all_targets,
        all_predictions,
        error,
        TorchUtils.to_numpy(running_loss),
        results_dir,
        name=label + "_error",
    )
    return torch.sqrt(running_loss)


def to_device(inputs: torch.Tensor) -> torch.Tensor:
    """
    Copies input tensors from CPU to GPU device for processing. GPU allows multithreading
    which makes computation faster. The rule of thumb is using 4 worker threads per GPU.
    See - https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor which needs to be loaded into GPU device.

    Returns
    -------
    torch.Tensor
        Input tensor allocated to GPU device.
    """
    if inputs.device != TorchUtils.get_device():
        inputs = inputs.to(device=TorchUtils.get_device())
    return inputs