import os
import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.model.data.inputs.data_handler import get_training_data
from bspysmg.model.data.outputs import test_model
from bspyalgo.utils.io import load_configs
from bspyproc.utils.pytorch import TorchUtils
from bspysmg.model.data.plots.model_results_plotter import plot_all
from bspyalgo.utils.io import save, create_directory

TorchUtils.force_cpu = True

configs = load_configs('./configs/training/test.json')

if 'seed' in configs:
    seed = configs['seed']
else:
    seed = None

seed = TorchUtils.init_seed(seed, deterministic=True)
configs['seed'] = seed

main_folder = 'training_data'

# # # # Get GD object with a processor specified in configs
model_generator = get_algorithm(configs, is_main=True)
model_generator.save_smg_configs_dict()

# # Get training and validation data
INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, INFO = get_training_data(model_generator.configs)
# Train the model
data = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), data_info=INFO)

results_dir = os.path.join(model_generator.base_dir, main_folder)
create_directory(results_dir)

train_targets = TorchUtils.get_numpy_from_tensor(TARGETS[:len(INPUTS_VAL)])
train_output = data.results['best_output_training']
plot_all(train_targets, train_output, results_dir, name='TRAINING')

val_targets = TorchUtils.get_numpy_from_tensor(TARGETS_VAL)
val_output = data.results['best_output']
plot_all(val_targets, val_output, results_dir, name='VALIDATION')

training_profile = data.results['performance_history'] * (model_generator.processor.get_amplification_value()**2)

plt.figure()
plt.plot(training_profile)
plt.title(f'Training profile')
plt.legend(['training', 'validation'])
plt.savefig(os.path.join(results_dir, 'training_profile'))

save('numpy', os.path.join(results_dir, 'training_summary.npz'), train_outputs=train_output, train_targets=train_targets, validation_outputs=val_output, validation_targets=val_targets)
# Test NN model with unseen test data

#TorchUtils.force_cpu = True

# TEST_ERROR = test_model.get_error(model_generator.dir_path + '/trained_network.pt',
#                                   model_generator.configs["data"]['test_data_path'])

test_model.get_error('tmp/output/model-02-2020/1ep_1e-4lr_128mb_2020_02_27_154800/reproducibility/model.pt',
                     model_generator.configs["data"]['test_data_path'], batch_size=128)
