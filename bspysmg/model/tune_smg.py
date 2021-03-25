from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from brainspy.utils.io import load_configs
from bspysmg.model.data.inputs.dataset import load_data
from bspysmg.model.data.outputs.train_model import generate_surrogate_model


def main(num_samples=10, max_num_epochs=15):
    # data_dir = os.path.abspath("./data")
    configs = load_configs("configs/training/smg_configs_template.yaml")
    # configs["processor"]["torch_model_dict"]["hidden_sizes"] =
    a = tune.choice([[90, 45, 25, 10], [45, 25], [45, 25], [90, 45, 25, 10]])
    dataloaders, amplification, data_info_dict = load_data(
        configs
    )  # Download data for all trials before starting the run

    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    result = tune.run(
        tune.with_parameters(generate_surrogate_model, configs=configs),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=configs,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    # model_state, optimizer_state = torch.load(checkpoint_path)
    # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main()