import os
import matplotlib.pyplot as plt
from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.model.data.inputs.data_handler import get_training_data
from bspyproc.utils.pytorch import TorchUtils
from bspysmg.model.data.plots.model_results_plotter import plot_all
from bspyalgo.utils.io import create_directory


def train_surrogate_model(configs, main_folder='training_data'):

    if 'seed' in configs:
        seed = configs['seed']
    else:
        seed = None

    seed = TorchUtils.init_seed(seed, deterministic=True)
    configs['seed'] = seed
    # Get training and validation data
    INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, INFO = get_training_data(configs)

    # Train the model
    model_generator = get_algorithm(configs, is_main=True)
    data = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), data_info=INFO)

    results_dir = os.path.join(model_generator.base_dir, main_folder)
    create_directory(results_dir)

    train_targets = INFO['processor']['amplification'] * TorchUtils.get_numpy_from_tensor(TARGETS[data.results['target_indices']][:len(INPUTS_VAL)])
    train_output = INFO['processor']['amplification'] * data.results['best_output_training']
    plot_all(train_targets, train_output, results_dir, name='TRAINING')

    val_targets = INFO['processor']['amplification'] * TorchUtils.get_numpy_from_tensor(TARGETS_VAL)
    val_output = INFO['processor']['amplification'] * data.results['best_output']
    plot_all(val_targets, val_output, results_dir, name='VALIDATION')

    training_profile = data.results['performance_history'] * (INFO['processor']['amplification']**2)

    plt.figure()
    plt.plot(training_profile)
    plt.title(f'Training profile')
    plt.legend(['training', 'validation'])
    plt.savefig(os.path.join(results_dir, 'training_profile'))

    model_generator.path_to_model = os.path.join(model_generator.base_dir, 'reproducibility', 'model.pt')
    return model_generator


if __name__ == "__main__":

    from bspyalgo.utils.io import load_configs

    configs = load_configs('configs/training/smg_configs_template.json')

    model_generator = train_surrogate_model(configs)

    print(f'Model saved in {model_generator.path_to_model}')
