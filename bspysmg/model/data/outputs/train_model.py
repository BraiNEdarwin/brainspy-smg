import os
import torch
import matplotlib.pyplot as plt
# from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.model.data.inputs.data_handler import get_training_data
from bspyproc.utils.pytorch import TorchUtils
from bspysmg.model.data.plots.model_results_plotter import plot_all
from bspyalgo.utils.io import create_directory
from bspysmg.model.data.inputs.dataset import ModelDataset
from bspyalgo.algorithms.gradient.fitter import split, train
from bspyalgo.utils.io import create_directory_timestamp


def train_surrogate_model(configs, model, criterion, optimizer, logger=None, main_folder='training_data'):

    results_dir = create_directory_timestamp(configs['results_base_dir'], main_folder)
    if 'seed' in configs:
        seed = configs['seed']
    else:
        seed = None

    seed = TorchUtils.init_seed(seed, deterministic=True)
    configs['seed'] = seed
    # Get training and validation data
    # INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, INFO = get_training_data(configs)
    dataset = ModelDataset(configs)
    amplification = dataset.info_dict['processor']['amplification']
    dataloaders, _ = split(dataset, configs['hyperparameters']['batch_size'], num_workers=configs['hyperparameters']['worker_no'], split_percentages=configs['hyperparameters']['split_percentages'])

    # Train the model
    model, performances = train(model, (dataloaders[0], dataloaders[1]), configs['hyperparameters']['epochs'], criterion, optimizer, logger=logger, save_dir=results_dir)
    # model_generator = get_algorithm(configs, is_main=True)
    # data = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), data_info=INFO)

    if len(dataloaders[1]) > 0:
        postprocess(dataloaders[0].dataset[dataloaders[0].sampler.indices][:len(dataloaders[1])], model, amplification, criterion, results_dir, label='TRAINING')
        postprocess(dataloaders[1].dataset[dataloaders[1].sampler.indices], model, amplification, criterion, results_dir, label='VALIDATION')
    else:
        # Default training evaluation 1000 values
        postprocess(dataloaders[0].dataset[dataloaders[0].sampler.indices][:1000], model, amplification, criterion, results_dir, label='TRAINING')
    if len(dataloaders[2]) > 0:
        postprocess(dataloaders[2].dataset[dataloaders[2].sampler.indices], model, amplification, criterion, cresults_dir, label='TEST')
    # train_targets = amplification * TorchUtils.get_numpy_from_tensor(TARGETS[data.results['target_indices']][:len(INPUTS_VAL)])
    # train_output = amplification * data.results['best_output_training']
    # plot_all(train_targets, train_output, results_dir, name='TRAINING')

    # val_targets = amplification * TorchUtils.get_numpy_from_tensor(TARGETS_VAL)
    # val_output = amplification * data.results['best_output']
    # plot_all(val_targets, val_output, results_dir, name='VALIDATION')

    training_profile = [TorchUtils.get_numpy_from_tensor(performances[i]) * (amplification ** 2) for i in range(len(performances))]

    plt.figure()
    plt.plot(training_profile)
    plt.title(f'Training profile')
    plt.legend(['training', 'validation'])
    plt.savefig(os.path.join(results_dir, 'training_profile'))

    #model_generator.path_to_model = os.path.join(model_generator.base_dir, 'reproducibility', 'model.pt')
    print('Model saved in :' + results_dir)
    # return model_generator


def postprocess(dataset, model, amplification, criterion, results_dir, label):

    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        predictions = model(inputs)

    train_targets = amplification * TorchUtils.get_numpy_from_tensor(targets)
    train_output = amplification * TorchUtils.get_numpy_from_tensor(predictions)
    plot_all(train_targets, train_output, results_dir, name=label)


if __name__ == "__main__":
    from bspyproc.processors.simulation.network import NeuralNetworkModel
    from bspyalgo.utils.io import load_configs

    configs = load_configs('configs/training/smg_configs_template.json')

    model = NeuralNetworkModel(configs['processor'])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['hyperparameters']['learning_rate'])
    criterion = torch.nn.MSELoss()
    train_surrogate_model(configs, model, criterion, optimizer)

    #print(f'Model saved in {model_generator.path_to_model}')
