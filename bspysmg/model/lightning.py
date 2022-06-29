from distutils.sysconfig import customize_compiler
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
#from __future__ import print_function

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


class TrainingModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.DataLoader],
        criterion: torch.nn.MSELoss,
        custom_optimizer: torch.optim.Optimizer,
        amplification: float,
        learning_rate: float = 1e-3,
    ) -> None:

        super(TrainingModel, self).__init__()
        self.name = 'custom'
        self.model = model
        self.dataloaders = dataloaders
        self.custom_opt = custom_optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        #self.amplification = amplification
        self.register_buffer('amplification', amplification)
        self.register_buffer('train_losses', torch.Tensor([]))
        self.register_buffer('val_losses', torch.Tensor([]))
        # self.train_losses, self.val_losses = TorchUtils.format(
        #     []), TorchUtils.format([])
        #self.train_losses, self.val_losses = torch.Tensor([]), torch.Tensor([])
        self.description = ""

    def configure_optimizers(self):
        opt = self.custom_opt(filter(lambda p: p.requires_grad,
                                     self.parameters()),
                              lr=self.learning_rate,
                              betas=(0.9, 0.75))
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
        if len(loss.shape) == 0:
            loss = loss.unsqueeze(0)
        return loss

    def training_step(self, batch, batch_idx):
        self.description = ""
        loss = self.shared_step(batch)

        running_loss = torch.sqrt(loss)
        running_loss *= self.amplification
        self.train_losses = torch.cat(
            (self.train_losses, running_loss.unsqueeze(dim=0)), dim=0)
        self.description = "Training loss (RMSE): {:.6f} (nA)\n".format(
            self.train_losses[-1].item())
        self.log('train_loss(nA)', running_loss, prog_bar=True)
        self.log('loss_epoch', {'train': loss}, on_step=False, on_epoch=True)
        self.log('loss_epoch_rmse_na', {'train': running_loss},
                 on_step=False,
                 on_epoch=True)

        return loss

    def forward(self, x):
        x = self.model(x)
        return x

    def validation_step(self, batch, batch_idx):
        if self.dataloaders[1] is not None and len(self.dataloaders[1]) > 0:
            loss = self.shared_step(batch)

            val_loss = torch.sqrt(loss)
            val_loss *= self.amplification
            self.val_losses = torch.cat(
                (self.val_losses, val_loss.unsqueeze(dim=0)), dim=0)
            # self.description += "Validation loss (RMSE): {:.6f} (nA)\n".format(
            #     self.val_losses[-1].item())
            #self.log(self.description)
            self.log('loss_epoch', {'val': loss}, on_step=False, on_epoch=True)
            self.log('loss_epoch_rmse_na', {'val': val_loss},
                     on_step=False,
                     on_epoch=True)
            self.log('val_loss(nA)', val_loss, prog_bar=True)
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
    model = TorchUtils.format(model)
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
    from pytorch_lightning.callbacks import QuantizationAwareTraining

    configs = load_configs("configs/training/smg_configs_template.yaml")
    custom_model = NeuralNetworkModel
    criterion = torch.nn.MSELoss()
    main_folder = 'training_folder'
    custom_optimizer = torch.optim.Adam

    init_seed(configs)
    results_dir = create_directory_timestamp(configs["results_base_dir"],
                                             main_folder)

    # Get training, validation and test data
    # Get amplification of the device and the info
    dataloaders, amplification, info_dict = get_dataloaders(configs)
    #amplification = TorchUtils.format(amplification)
    # Initilialise model
    model = custom_model(info_dict["model_structure"])
    # model.set_info_dict(info_dict)
    #model = TorchUtils.format(model)

    model_lightning = TrainingModel(
        model, dataloaders, criterion, custom_optimizer, amplification,
        configs["hyperparameters"]["learning_rate"])

    checkpoint_callback = ModelCheckpoint(monitor='val_loss(nA)',
                                          filename='sample-{val_acc:.3f}',
                                          mode='min')
    #earlystopping_callback = EarlyStopping(monitor="val_loss(nA)", mode="min")

    trainer = pl.Trainer(gpus=1,
                         max_epochs=configs["hyperparameters"]["epochs"],
                         callbacks=[checkpoint_callback]
                         #, resume_from_checkpoint=''
                         )

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
