
from bspyalgo.utils.io import load_configs
from bspysmg.model.data.outputs import test_model
from bspysmg.model.data.outputs.train_model import train_surrogate_model

configs = load_configs('configs/training/smg_configs_template.json')

model_generator = train_surrogate_model(configs)

# Test NN model with unseen test data
mse = test_model.get_error(model_generator.path_to_model, model_generator.configs["data"]['test_data_path'])
print(f'MSE: {mse}')
