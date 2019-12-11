import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.model.data.inputs.data_handler import get_training_data
from bspysmg.model.data.outputs import test_model

# Get GD object with a processor specified in configs
model_generator = get_algorithm('./configs/training/smg_configs_template.json')

# Get training and validation data
INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, INFO = get_training_data(model_generator.configs)
# Train the model
DATA = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), data_info=INFO)
LOSS = DATA.results['performance_history'] * (model_generator.processor.get_amplification_value()**2)

plt.figure()
plt.plot(LOSS)
plt.title(f'Training profile')
plt.legend(['training', 'validation'])
plt.savefig(model_generator.dir_path + '/training_profile')

np.savez(model_generator.dir_path + '/training_profile.npz', LOSS=LOSS)
# Test NN model with unseen test data
TEST_ERROR = test_model.get_error(model_generator.dir_path + '/trained_network.pt',
                                  model_generator.configs["data"]['test_data_path'])
