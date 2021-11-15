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


def init_seed(configs : dict) -> None:
    """
    Initializes a random seed for training. A random seed is a starting point for pseudorandom
    number generator algorithms which is used for reproducibility.
    Also see - https://pytorch.org/docs/stable/notes/randomness.html  

    Parameters
    ----------
    configs : dict
         Training configurations with the following keys:

        - seed:  int [Optional]
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
        configs : dict,
        custom_model : torch.nn.Module = NeuralNetworkModel,
        criterion : torch.nn = MSELoss(),
        custom_optimizer : torch.optim = Adam,
        main_folder : str = "training_data",
) -> None:
    """
    Initialises a neural network model, trains it and plots the training,
    validation and testing loss curves. It also saves the model and results
    to a specified dicrectory.

    Parameters
    ----------
    configs : dict
        Training configurations for training a model.
    custom_model : custom NeuralNetworkModel of type torch.nn.Module
        Model to be trained.
    criterion : <method>
        Fitness/loss function that will be used to train the model.
    custom_optimizer : torch.optim
        Optimization method used to train the model which decreases model's loss.
    save_dir : string [Optional]
        Name of the path where the trained model is to be saved.
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
    model, performances = train_loop(
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
    if dataloaders[-1].dataset.tag == 'test':
        plt.plot(np.ones(len(performances[-1])) * TorchUtils.to_numpy(loss))
        plt.title("Training profile /n Test loss : %.6f (nA)" % loss)
    else:
        plt.title("Training profile")
    if not len(performances[1]) == 0:
        plt.legend(["training", "validation"])
    plt.xlabel("Epoch no.")
    plt.ylabel("RMSE (nA)")
    plt.savefig(os.path.join(results_dir, "training_profile"))
    if not dataloaders[-1].dataset.tag == 'train':
        training_data = torch.load(
            os.path.join(results_dir, "training_data.pt"))
        training_data['test_loss'] = loss
        torch.save(training_data, os.path.join(results_dir,
                                               "training_data.pt"))
    # print("Model saved in :" + results_dir)


def train_loop(
    model : torch.nn.Module,
    info_dict : dict,
    dataloaders : List[torch.utils.data.DataLoader],
    criterion : torch.nn.modules.loss,
    optimizer : torch.optim,
    epochs : int,
    amplification : float,
    start_epoch : int = 0,
    save_dir : str = None,
    early_stopping : bool = True,
) -> Tuple[torch.nn.Module, List[float]]:
    """
    Performs the training of a model and returns the trained model, training loss
    validation loss.

    Parameters
    ----------
    model : custom NeuralNetworkModel of type torch.nn.Module
        Model to be trained.
    info_dict : dict
        The dictionary used for initialising the surrogate model.
    dataloaders :  list
        A list containing a single PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Fitness/loss function that will be used to train the model.
    optimizer : torch.optim
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
    tuple
        Trained model and a list of training loss and validation loss.
    """
    if start_epoch > 0:
        start_epoch += 1

    train_losses, val_losses = TorchUtils.format([]), TorchUtils.format([])
    min_val_loss = np.inf

    for epoch in range(epochs):
        print("\nEpoch: " + str(epoch))
        model, running_loss = default_train_step(model, dataloaders[0],
                                                 criterion, optimizer)
        running_loss *= amplification
        running_loss = torch.sqrt(running_loss)
        train_losses = torch.cat((train_losses, running_loss.unsqueeze(dim=0)),
                                 dim=0)
        description = "Training loss (RMSE): {:.6f} (nA)\n".format(
            train_losses[-1].item())

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            val_loss = default_val_step(model, dataloaders[1], criterion)
            val_loss *= amplification
            val_loss = torch.sqrt(val_loss)
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

    return model, [train_losses, val_losses]


def default_train_step(model : torch.nn.Module,
dataloader : torch.utils.data.DataLoader,
criterion : torch.nn.modules.loss,
optimizer : torch.optim
) -> Tuple[torch.nn.Module, float]:
    """
    Performs the training step of a model within a single epoch and returns the
    current loss and current trained model.

    Parameters
    ----------
    model : custom NeuralNetworkModel of type torch.nn.Module
        Model to be trained.
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Fitness/loss function that will be used to train the model.
    optimizer : torch.optim
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


def default_val_step(model : torch.nn.Module,
dataloader : torch.utils.data.DataLoader,
criterion : torch.nn.modules.loss
) -> float:
    """
    Performs the validation step of a model within a single epoch and returns
    the validation loss.

    Parameters
    ----------
    model : custom NeuralNetworkModel of type torch.nn.Module
        Model to be trained.
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    criterion : <method>
        Fitness/loss function that will be used to train the model.

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


def postprocess(dataloader : torch.utils.data.DataLoader,
model : torch.nn.Module,
criterion : torch.nn.modules.loss,
amplification : float,
results_dir : str,
label : str
) -> float:
    """
    Plots error vs output and error histogram for given dataset and saves it to
    specified directory.

    Parameters
    ----------
    dataloader :  torch.utils.data.DataLoader
        A PyTorch Dataloader containing the training dataset.
    model : custom NeuralNetworkModel of type torch.nn.Module
        Model to be trained.
    criterion : <method>
        Fitness/loss function that will be used to train the model.
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
    running_loss = running_loss * amplification

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


def to_device(inputs : torch.Tensor) -> torch.Tensor:
    """
    Copies input tensors to current device for processing.
    See - https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor which needs to be loaded into current device.

    Returns
    -------
    tuple
        Input tensor allocated to current device.
    """
    if inputs.device != TorchUtils.get_device():
        inputs = inputs.to(device=TorchUtils.get_device())
    return inputs


if __name__ == "__main__":
    from brainspy.utils.io import load_configs

    configs = load_configs("configs/training/smg_configs_template.yaml")

    generate_surrogate_model(configs)
