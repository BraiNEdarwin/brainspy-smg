from bspyalgo.algorithm_manager import get_algorithm
from bspysmg.measurement.data.inputs import get_nn_data
from bspysmg.measurement.data.outputs.core import test_model

# Get GD object with a processor specified in configs
model_generator = get_algorithm('./configs/training/smg_configs_template.json')

# Get training and validation data
INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL = get_nn_data(model_generator.configs)

# Train the model
# TODO: maybe the sets are being duplicted?
DATA = model_generator.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL))

# Test NN model with unseen test data
TEST_ERROR = test_model.get_error(model_generator)

# Predict Control Voltages
control_voltages = test_model.get_control_voltages(model_generator.configs)
