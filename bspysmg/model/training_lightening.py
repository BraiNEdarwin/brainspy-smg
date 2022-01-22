import os
import torch
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.io import create_directory_timestamp
from brainspy.processors.simulation.model import NeuralNetworkModel
from bspysmg.data.dataset import get_dataloaders
from bspysmg.utils.plots import plot_error_vs_output, plot_error_hist
from typing import Tuple, List
from __future__ import print_function

import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def init_seed(configs: dict) -> None:
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
        seed = seed_everything(configs['seed'], workers=True)
    else:
        seed = seed_everything(workers=True)
    configs["seed"] = seed


class SurrogateModel(pl.LightningModule):
    def __init__(self, 
                model: torch.nn.Module,
                dataloaders: List[torch.utils.data.Dataloader],
                criterion: torch.nn.MSELoss,
                custom_optimizer: torch.optim.Optimizer,
                amplification: float,
                learning_rate: float = 1e-3,
                ) -> None:

        super(SurrogateModel, self).__init__()
        self.name = 'custom'
        self.model = model
        self.dataloaders = dataloaders
        self.custom_opt = custom_optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.amplification = amplification
        self.train_losses, self.val_losses = TorchUtils.format([]), TorchUtils.format([])
        self.description = ""

    def configure_optimizers(self):
        opt = self.custom_opt(filter(lambda p: p.requires_grad, self.parameters()),
                            lr=self.learning_rate,
                            betas=(0.9, 0.75)
                            )
        return opt

    def train_dataloader(self):
        return self.dataloaders[0]

    def val_dataloader(self):
        return self.dataloaders[1]

    def test_dataloader(self):
        return self.dataloaders[2]

    def shared_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        return loss

    def training_step(self, batch, batch_idx):
        self.description = ""
        loss = self.shared_step(batch)
        self.log('loss_epoch', {'train': loss}, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=False)

        running_loss = torch.sqrt(loss)
        running_loss *= self.amplification
        self.train_losses = torch.cat((self.train_losses, running_loss.unsqueeze(dim=0)),
                                 dim=0)
        self.description = "Training loss (RMSE): {:.6f} (nA)\n".format(
            self.train_losses[-1].item())
        return loss

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step_end(self, outputs):
        self.model.constraint_weights()
        return outputs

    def validation_step(self, batch, batch_idx):
        if self.dataloaders[1] is not None and len(self.dataloaders[1]) > 0:
            loss = self.shared_step(batch)
            self.log('loss_epoch', {'val': loss}, on_step=False, on_epoch=True)
            self.log('val_loss', loss, prog_bar=True)
            val_loss = torch.sqrt(loss)
            val_loss *= self.amplification
            self.val_losses = torch.cat((self.val_losses, val_loss.unsqueeze(dim=0)),
                                dim=0)
            self.description += "Validation loss (RMSE): {:.6f} (nA)\n".format(
                self.val_losses[-1].item())
            self.log(self.description)
            return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('loss_epoch', {'test': loss}, on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def get_performance(self):
        return [self.train_losses, self.val_losses]


def generate_surrogate_model(configs: dict,
                             custom_model: torch.nn.Module,
                             criterion: torch.nn.MSELoss,
                             custom_optimizer: torch.optim.Optimizer,
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
    custom_model : custom model of type torch.nn.Module
        Model to be trained.
    criterion : <method>
        Loss function that will be used to train the model.
    custom_optimizer : torch.optim.Optimizer
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

    model_lightning = SurrogateModel(model,
                                    dataloaders,
                                    criterion,
                                    custom_optimizer,
                                    amplification,
                                    configs["hyperparameters"]["learning_rate"]
                                    )

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                        filename='sample-{val_acc:.3f}',
                                        mode='min')
    earlystopping_callback = EarlyStopping(monitor="val_loss", mode="min")
    
    trainer = pl.Trainer(max_epochs=configs["hyperparameters"]["epochs"],
                        callbacks=[checkpoint_callback, earlystopping_callback])

    trainer.fit(model_lightning)
    performances = model_lightning.get_performance()

    # Plot results
    labels = ["TRAINING", "VALIDATION", "TEST"]
    for i in range(len(dataloaders)):
        if dataloaders[i] is not None:
            loss = postprocess(
                dataloaders[i],
                model_lightning,
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


def postprocess(dataloader: torch.utils.data.DataLoader,
                model: torch.nn.Module, criterion: torch.nn.MSELoss,
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


if __name__ == "__main__":
    from brainspy.utils.io import load_configs

    configs = load_configs("configs/training/smg_configs_template.yaml")

    generate_surrogate_model(configs)
