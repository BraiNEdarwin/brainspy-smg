from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.measurement.data.inputs.data_handler import get_training_data
from bspysmg.measurement.data.outputs.core import test_model

# Get GD object with a processor specified in configs
model_generator = get_algorithm('./configs/training/smg_configs_template.json')

# # Get training and validation data
# INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, INFO = get_training_data(model_generator.configs["data"])

# # Train the model
# TODO: implement the  dict entry of the model's location and make sure it is saved and all relations between data, models and configs
# DATA = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL))

# Test NN model with unseen test data
# TODO: remove when saving path is same as model path
model_generator.configs['path_to_model'] = r"/home/hruiz/Documents/PROJECTS/DARWIN/Code/packages/brainspy/brainspy-processors/tmp/input/models/nn_test/checkpoint3000_02-07-23h47m.pt"
TEST_ERROR = test_model.get_error(model_generator.configs['path_to_model'],
                                  model_generator.configs["data"]['test_data_path'],
                                  steps=3)

# Predict Control Voltages
# TODO: implement
CONTROL_VOLTAGES = test_model.get_control_voltages(model_generator.configs["validation_task"])
