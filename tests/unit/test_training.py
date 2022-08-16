import unittest
from bspysmg.model import training
from brainspy.utils.io import create_directory_timestamp
from bspysmg.data.dataset import get_dataloaders
from brainspy.processors.simulation.model import NeuralNetworkModel
from brainspy.utils.pytorch import TorchUtils
from torch.optim import Adam
from torch.nn import MSELoss


class Test_Training(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(Test_Training, self).__init__(*args, **kwargs)
        self.configs = {'results_base_dir': 'tests/data'}
        self.configs['model_structure'] = {
            'hidden_sizes': [90] * 5,
            'D_in': 7,
            'D_out': 1,
            'activation': 'relu'
        }
        self.configs['hyperparameters'] = {
            'epochs': 2,
            'learning_rate': 1.0e-05
        }
        self.configs['data'] = {
            'dataset_paths': [
                "tests/data/postprocessed_data.npz",
                "tests/data/postprocessed_data.npz",
                "tests/data/postprocessed_data.npz"
            ],
            'steps':
            4,
            'batch_size':
            128,
            'worker_no':
            0,
            'pin_memory':
            False,
            'split_percentages': [1, 0]
        }

    def test_init_seed(self):
        with self.assertRaises(TypeError):
            training.init_seed()

        try:
            training.init_seed(self.configs)
        except Exception:
            self.fail("Failed Execution: init_seed()")

        with self.assertRaises(TypeError):
            training.init_seed([])

    def test_surrogate_model(self):
        with self.assertRaises(TypeError):
            training.generate_surrogate_model()

        training.generate_surrogate_model(self.configs, main_folder='tests/data')

    def test_train_loop(self):

        try:
            training.init_seed(self.configs)
            results_dir = create_directory_timestamp(
                self.configs["results_base_dir"], 'tests/data')

            dataloaders, amplification, info_dict = get_dataloaders(
                self.configs)

            model = NeuralNetworkModel(info_dict["model_structure"])
            model = TorchUtils.format(model)

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.configs["hyperparameters"]["learning_rate"],
                betas=(0.9, 0.75),
            )

            model, performances,_ = training.train_loop(
                model,
                info_dict, (dataloaders[0], dataloaders[1]),
                MSELoss(),
                optimizer,
                self.configs["hyperparameters"]["epochs"],
                amplification,
                save_dir=results_dir)
        except Exception:
            self.fail("Failed Execution: train_loop()")

    def test_train_loop_no_early_stopping(self):

        try:
            training.init_seed(self.configs)
            results_dir = create_directory_timestamp(
                self.configs["results_base_dir"], 'tests/data')

            dataloaders, amplification, info_dict = get_dataloaders(
                self.configs)

            model = NeuralNetworkModel(info_dict["model_structure"])
            model = TorchUtils.format(model)

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.configs["hyperparameters"]["learning_rate"],
                betas=(0.9, 0.75),
            )

            model, performances, _ = training.train_loop(
                model,
                info_dict, (dataloaders[0], None, None),
                MSELoss(),
                optimizer,
                self.configs["hyperparameters"]["epochs"],
                amplification,
                early_stopping=False,
                save_dir=results_dir)
        except Exception:
            self.fail("Failed Execution: train_loop()")

    def test_train_step(self):

        try:
            training.init_seed(self.configs)

            dataloaders, amplification, info_dict = get_dataloaders(
                self.configs)

            model = NeuralNetworkModel(info_dict["model_structure"])
            model = TorchUtils.format(model)

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.configs["hyperparameters"]["learning_rate"],
                betas=(0.9, 0.75),
            )

            model, running_loss = training.default_train_step(
                model, dataloaders[0], MSELoss(), optimizer)
        except:
            self.fail("Failed Execution: train_step()")

    def test_val_step(self):

        try:
            training.init_seed(self.configs)

            dataloaders, amplification, info_dict = get_dataloaders(
                self.configs)

            model = NeuralNetworkModel(info_dict["model_structure"])
            model = TorchUtils.format(model)

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.configs["hyperparameters"]["learning_rate"],
                betas=(0.9, 0.75),
            )

            val_loss = training.default_val_step(model, dataloaders[1],
                                                 MSELoss())
        except:
            self.fail("Failed Execution: val_step()")

    def test_to_device(self):
        try:
            dataloaders, amplification, info_dict = get_dataloaders(
                self.configs)
            for input, target in dataloaders[0]:
                input, target = training.to_device(input), training.to_device(
                    target)
        except:
            self.fail("Failed Execution: to_device()")

    def test_postprocess(self):
        try:
            training.init_seed(self.configs)
            results_dir = create_directory_timestamp(
                self.configs["results_base_dir"], 'tests/data')

            dataloaders, amplification, info_dict = get_dataloaders(
                self.configs)

            model = NeuralNetworkModel(info_dict["model_structure"])
            model = TorchUtils.format(model)

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.configs["hyperparameters"]["learning_rate"],
                betas=(0.9, 0.75),
            )

            model, performances, _ = training.train_loop(
                model,
                info_dict,
                (dataloaders[0], dataloaders[1]),
                MSELoss(),
                optimizer,
                self.configs["hyperparameters"]["epochs"],
                amplification,
                save_dir=results_dir,
            )
            labels = ["TRAINING", "VALIDATION", "TEST"]
            for i in range(len(dataloaders)):
                if dataloaders[i] is not None:
                    loss = training.postprocess(
                        dataloaders[i],
                        model,
                        MSELoss(),
                        amplification,
                        results_dir,
                        label=labels[i],
                    )
        except:
            self.fail("Failed Execution: postprocess()")


if __name__ == "__main__":
    unittest.main()