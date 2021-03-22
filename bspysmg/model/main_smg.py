import torch
from brainspy.processors.simulation.model import NeuralNetworkModel
from brainspy.utils.io import load_configs
from brainspy.utils.pytorch import TorchUtils

from bspysmg.model.data.outputs.train_model import train_surrogate_model

# TorchUtils.force_cpu = True

CONFIGS = load_configs("configs/training/smg_configs_template.yaml")

MODEL = NeuralNetworkModel()
MODEL.build_model_structure(CONFIGS["processor"]["torch_model_dict"])

OPTIMIZER = torch.optim.Adam(
    filter(lambda p: p.requires_grad, MODEL.parameters()),
    lr=CONFIGS["hyperparameters"]["learning_rate"],
)
CRITERION = torch.nn.MSELoss()
train_surrogate_model(CONFIGS, MODEL, CRITERION, OPTIMIZER)
