
import torch
from bspyproc.processors.simulation.network import NeuralNetworkModel
from bspyalgo.utils.io import load_configs
from bspysmg.model.data.outputs import test_model
from bspysmg.model.data.outputs.train_model import train_surrogate_model
from bspyproc.utils.pytorch import TorchUtils

TorchUtils.force_cpu = True

configs = load_configs('configs/training/smg_configs_template.json')

model = NeuralNetworkModel(configs['processor'])
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['hyperparameters']['learning_rate'])
criterion = torch.nn.MSELoss()
train_surrogate_model(configs, model, criterion, optimizer)

# # Test NN model with unseen test data
# mse = test_model.get_error(model_generator.path_to_model, model_generator.configs["data"]['test_data_path'])
# print(f'MSE: {mse}')
