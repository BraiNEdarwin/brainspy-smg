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
from bspysmg.model.data.inputs.dataset import load_data
from bspysmg.model.data.plots.model_results_plotter import plot_all


def init_seed(configs):
    if "seed" in configs:
        seed = configs["seed"]
    else:
        seed = None

    seed = TorchUtils.init_seed(seed, deterministic=True)
    configs["seed"] = seed


def generate_surrogate_model(
    configs,
    custom_model=NeuralNetworkModel,
    criterion=MSELoss(),
    custom_optimizer=Adam,
    main_folder="training_data",
):
    print(configs)
    # Initialise seed and create data directories
    init_seed(configs)
    results_dir = create_directory_timestamp(configs["results_base_dir"], main_folder)

    # Get training, validation and test data
    # Get amplification of the device and the info
    dataloaders, amplification, info_dict = load_data(configs)

    # Initilialise model
    model = custom_model(configs["model_architecture"])
    model.set_info_dict(info_dict)
    model = TorchUtils.format_model(model)

    # Initialise optimiser
    optimizer = custom_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs["hyperparameters"]["learning_rate"],
    )

    # Whole training loop
    model, performances = train_loop(
        model,
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
            postprocess(
                dataloaders[i],
                model,
                criterion,
                amplification,
                results_dir,
                label=labels[i],
            )

    test_loss = None
    if dataloaders[2] is not None:
        test_loss = default_val_step(model, dataloaders[2], criterion, amplification)
        print("Test loss: " + str(test_loss))

    performances = np.array(performances)
    plt.figure()
    plt.plot(performances[0])
    if not performances[1] == []:
        plt.plot(performances[1])
    if test_loss is None:
        plt.title("Training profile")
    else:
        plt.title(
            "Training profile (Amplified)/n Amplified Test loss: %.8f" % test_loss
        )
    if not performances[1] == []:
        plt.legend(["training", "validation"])
    plt.savefig(os.path.join(results_dir, "training_profile"))

    # print("Model saved in :" + results_dir)


def train_loop(
    model,
    dataloaders,
    criterion,
    optimizer,
    epochs,
    amplification,
    start_epoch=0,
    save_dir=None,
    early_stopping=True,
):
    if start_epoch > 0:
        start_epoch += 1

    train_losses, val_losses = [], []
    min_val_loss = np.inf

    for epoch in range(epochs):
        print("\nEpoch: " + str(epoch))
        model, running_loss = default_train_step(
            model, dataloaders[0], criterion, optimizer, amplification
        )
        train_losses.append(running_loss)
        description = "Amplified training loss: {:.8f} \n".format(train_losses[-1])

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            val_loss = default_val_step(model, dataloaders[1], criterion, amplification)
            val_losses.append(val_loss)
            description += "Amplified validation loss: {:.8f} \n".format(val_losses[-1])
            # Save only when peak val performance is reached
            if (
                save_dir is not None
                and early_stopping
                and val_losses[-1] < min_val_loss
            ):
                min_val_loss = val_losses[-1]
                description += "Model saved in: " + save_dir
                torch.save(model, os.path.join(save_dir, "model.pt"))
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "min_val_loss": min_val_loss,
                    },
                    os.path.join(save_dir, "training_data.pickle"),
                )

        print(description)
        # looper.set_description(description)

    # TODO: Add a save instruction and a stopping criteria
    # if stopping_criteria(train_losses, val_losses):
    #     break
    print("Finished training model. ")
    print("Model saved in: " + save_dir)
    if (
        save_dir is not None
        and early_stopping
        and dataloaders[1] is not None
        and len(dataloaders[1]) > 0
    ):
        model = torch.load(os.path.join(save_dir, "model.pt"))
        print("Amplified validation loss: " + str(min_val_loss))
    else:
        torch.save(model, os.path.join(save_dir, "model.pt"))

    return model, [train_losses, val_losses]


def default_train_step(model, dataloader, criterion, optimizer, amplification):
    running_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader):
        inputs, targets = to_device(inputs, targets)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        running_loss += amplification * loss.item() * inputs.shape[0]
    running_loss /= len(dataloader.dataset)
    return model, running_loss


def default_val_step(model, dataloader, criterion, amplification):
    with torch.no_grad():
        val_loss = 0
        model.eval()
        for inputs, targets in tqdm(dataloader):
            inputs, targets = to_device(inputs, targets)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            val_loss += amplification * loss.item() * inputs.shape[0]
        val_loss /= len(dataloader.dataset)
    return val_loss


def postprocess(dataloader, model, criterion, amplification, results_dir, label):
    print(f"Postprocessing {label} data ... ")
    # i = 0
    running_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for inputs, targets in tqdm(dataloader):
            inputs, targets = to_device(inputs, targets)
            all_targets.append(amplification * targets.squeeze())
            all_predictions.append(amplification * model(inputs).squeeze())
            loss = criterion(all_predictions[-1], all_targets[-1])
            running_loss += loss.item() * inputs.shape[0]  # sum up batch loss

    running_loss /= len(dataloader.dataset)
    print(str(criterion) + ": " + str(running_loss))

    all_targets = TorchUtils.get_numpy_from_tensor(torch.cat(all_targets))
    all_predictions = TorchUtils.get_numpy_from_tensor(torch.cat(all_predictions))

    plot_all(all_targets, all_predictions, results_dir, name=label)


def to_device(inputs, targets):
    if inputs.device != TorchUtils.get_accelerator_type():
        inputs = inputs.to(device=TorchUtils.get_accelerator_type())
    if targets.device != TorchUtils.get_accelerator_type():
        targets = targets.to(device=TorchUtils.get_accelerator_type())
    return (inputs, targets)
