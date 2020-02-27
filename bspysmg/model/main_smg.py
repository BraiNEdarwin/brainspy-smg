import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.model.data.inputs.data_handler import get_training_data
from bspysmg.model.data.outputs import test_model
from bspyalgo.utils.io import load_configs
from bspyproc.utils.pytorch import TorchUtils

TorchUtils.force_cpu = True

configs = load_configs('./configs/training/test.json')

if 'seed' in configs:
    seed = configs['seed']
else:
    seed = None

seed = TorchUtils.init_seed(seed, deterministic=True)
configs['seed'] = seed

# # # # Get GD object with a processor specified in configs
model_generator = get_algorithm(configs, is_main=True)
model_generator.save_smg_configs_dict()

# Get training and validation data
INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, INFO = get_training_data(model_generator.configs)
# Train the model
DATA = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), data_info=INFO)
LOSS = DATA.results['performance_history'] * (model_generator.processor.get_amplification_value()**2)

plt.figure()
plt.plot(LOSS)
plt.title(f'Training profile')
plt.legend(['training', 'validation'])
plt.savefig(configs['results_base_dir'] + '/training_profile')

np.savez(configs['results_base_dir'] + '/summary.npz', LOSS=LOSS)
# Test NN model with unseen test data

#TorchUtils.force_cpu = True

# TEST_ERROR = test_model.get_error(model_generator.dir_path + '/trained_network.pt',
#                                   model_generator.configs["data"]['test_data_path'])

TEST_ERROR = test_model.get_error('tmp/output/model2020/1ep_1e-4lr_128mb_2020_02_13_180627/trained_network.pt',
                                  model_generator.configs["data"]['test_data_path'])

model_generator.save_model_mse(TEST_ERROR)
