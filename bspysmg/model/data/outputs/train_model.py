import os
import torch
import matplotlib.pyplot as plt

# from brainspy.algorithm_manager import get_algorithm
# from bspysmg.model.data.inputs.data_handler import get_training_data
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.io import create_directory_timestamp
from brainspy.algorithms.gd import train

from bspysmg.model.data.inputs.dataset import load_data
from bspysmg.model.data.plots.model_results_plotter import plot_all


def train_surrogate_model(
    configs, model, criterion, optimizer, logger=None, main_folder="training_data"
):

    results_dir = create_directory_timestamp(configs["results_base_dir"], main_folder)
    if "seed" in configs:
        seed = configs["seed"]
    else:
        seed = None

    seed = TorchUtils.init_seed(seed, deterministic=True)
    configs["seed"] = seed
    # Get training and validation data
    # INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, INFO = get_training_data(configs)

    dataloaders, amplification, data_info_dict = load_data(configs)
    info_dict = {}
    info_dict["data_info"] = data_info_dict
    info_dict["smg_configs"] = configs
    model.set_info_dict(info_dict)
    model, performances = train(
        model,
        (dataloaders[0], dataloaders[1]),
        criterion,
        optimizer,
        configs["hyperparameters"],
        logger=logger,
        save_dir=results_dir,
    )
    # model_generator = get_algorithm(configs, is_main=True)
    # data = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), data_info=INFO)
    labels = ["TRAINING", "VALIDATION", "TEST"]
    for i in range(len(dataloaders)):
        if dataloaders[i] is not None:
            postprocess(
                dataloaders[i], model, amplification, results_dir, label=labels[i]
            )

    # train_targets = amplification * TorchUtils.get_numpy_from_tensor(TARGETS[data.results['target_indices']][:len(INPUTS_VAL)])
    # train_output = amplification * data.results['best_output_training']
    # plot_all(train_targets, train_output, results_dir, name='TRAINING')

    # val_targets = amplification * TorchUtils.get_numpy_from_tensor(TARGETS_VAL)
    # val_output = amplification * data.results['best_output']
    # plot_all(val_targets, val_output, results_dir, name='VALIDATION')

    training_profile = [
        TorchUtils.get_numpy_from_tensor(performances["performance_history"][i])
        * (amplification ** 2)
        for i in range(len(performances["performance_history"]))
    ]

    plt.figure()
    for i in range(len(training_profile)):
        plt.plot(training_profile[i])
    plt.title(f"Training profile")
    plt.legend(["training", "validation"])
    plt.savefig(os.path.join(results_dir, "training_profile"))

    # # Save the model according to the SMG standard
    # state_dict = model.state_dict()
    # state_dict["info"] = {}
    # state_dict["info"]["data_info"] = data_info_dict
    # state_dict["info"]["smg_configs"] = configs
    # torch.save(state_dict, os.path.join(results_dir, "model.pt"))

    # model_generator.path_to_model = os.path.join(model_generator.base_dir, 'reproducibility', 'model.pt')
    print("Model saved in :" + results_dir)
    # return model_generator


def postprocess(dataloader, model, amplification, results_dir, label):
    print(f"Postprocessing {label} data ... ")
    predictions = TorchUtils.format_tensor(
        torch.zeros(len(dataloader), dataloader.batch_size)
    )
    targets_log = TorchUtils.format_tensor(
        torch.zeros(len(dataloader), dataloader.batch_size)
    )
    i = 0
    with torch.no_grad():
        model.eval()
        for inputs, targets in dataloader:
            if inputs.device != TorchUtils.get_accelerator_type():
                inputs = inputs.to(device=TorchUtils.get_accelerator_type())
            if targets.device != TorchUtils.get_accelerator_type():
                targets = targets.to(device=TorchUtils.get_accelerator_type())
            targets_log[i] = targets.squeeze()
            predictions[i] = model(inputs).squeeze()
            i += 1
        # inputs, targets = dataset[:]
        # inputs = inputs.to(device=TorchUtils.get_accelerator_type())
        # targets = targets.to(device=TorchUtils.get_accelerator_type())
        # predictions = model(inputs)

    # train_targets = amplification * TorchUtils.get_numpy_from_tensor(targets_log)
    targets_log = targets_log.view(targets_log.shape[0] * targets_log.shape[1])
    predictions = predictions.view(predictions.shape[0] * predictions.shape[1])
    train_targets = amplification * TorchUtils.get_numpy_from_tensor(targets_log)
    train_output = amplification * TorchUtils.get_numpy_from_tensor(predictions)
    plot_all(train_targets, train_output, results_dir, name=label)
