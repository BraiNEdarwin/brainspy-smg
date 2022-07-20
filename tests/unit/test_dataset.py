import unittest
import torch
from bspysmg.data import dataset
from brainspy.utils.io import load_configs


class Test_Dataset(unittest.TestCase):
    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            dataset.ModelDataset()

    def test_dataset(self):
        try:
            ds = dataset.ModelDataset("tests/data/postprocessed_data.npz")
            dl = torch.utils.data.DataLoader(ds, batch_size=100)
            next(iter(dl))
            len(dl)
        except Exception:
            self.fail('Dataloader could not be used with the Model Dataset')

    def test_normal_inputs(self):
        try:
            dataset.ModelDataset("tests/data/postprocessed_data.npz", steps=10)
        except Exception:
            self.fail("Failed Execution: Steps - 10, Tag - train")
        try:
            dataset.ModelDataset("tests/data/postprocessed_data.npz", steps=5)
        except Exception:
            self.fail("Failed Execution: Steps - 5, Tag - validation")
        try:
            dataset.ModelDataset("tests/data/postprocessed_data.npz", steps=1)
        except Exception:
            self.fail("Failed Execution: Steps - 1, Tag - test")

    def test_check_file_exists(self):
        with self.assertRaises(FileNotFoundError):
            dataset.ModelDataset("test_fail.npz")

    def test_steps_values(self):
        with self.assertRaises(TypeError):
            dataset.ModelDataset("tests/data/postprocessed_data.npz",
                                 steps=0.1)
        with self.assertRaises(TypeError):
            dataset.ModelDataset("tests/data/postprocessed_data.npz",
                                 steps="10")
        with self.assertRaises(TypeError):
            dataset.ModelDataset("tests/data/postprocessed_data.npz",
                                 steps=[1, 2, 3])

    def test_get_dataloaders_three_paths(self):
        configs = load_configs("configs/training/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = [
            "tests/data/postprocessed_data.npz"
        ] * 3
        try:
            dataset.get_dataloaders(configs)
        except Exception:
            self.fail(
                "Execution Failed: for three dataset paths (train, val and test)"
            )

    def test_get_dataloaders_two_paths(self):
        configs = load_configs("tests/data/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = [
            "tests/data/postprocessed_data.npz"
        ] * 2
        try:
            dataset.get_dataloaders(configs)
        except Exception:
            self.fail(
                "Execution Failed: for two dataset paths (train and val)")

    def test_get_dataloaders_one_path(self):
        configs = load_configs("tests/data/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = [
            "tests/data/postprocessed_data.npz"
        ]
        try:
            dataset.get_dataloaders(configs)
        except Exception:
            self.fail("Execution Failed: for a single dataset path (train)")

    def test_get_dataloaders_one_path_one_split(self):
        configs = load_configs("tests/data/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = [
            "tests/data/postprocessed_data.npz"
        ]
        configs['data']['split_percentages'] = [1]
        try:
            dataset.get_dataloaders(configs)
        except Exception:
            self.fail("Execution Failed: for a single dataset path (train)")

    def test_get_dataloaders_one_path_two_split(self):
        configs = load_configs("tests/data/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = [
            "tests/data/postprocessed_data.npz"
        ]
        configs['data']['split_percentages'] = [0.8, 0.2]
        try:
            dataset.get_dataloaders(configs)
        except Exception:
            self.fail("Execution Failed: for a single dataset path (train)")

    def test_get_dataloaders_two_path_one_none(self):
        configs = load_configs("configs/training/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = [
            "tests/data/postprocessed_data.npz", None
        ]
        try:
            dataset.get_dataloaders(configs)
        except Exception:
            self.fail("Execution Failed: for a single dataset path (train)")

    def test_get_dataloaders_assertions(self):
        configs = load_configs("tests/data/smg_configs_template.yaml")
        with self.assertRaises(AssertionError):
            configs_1 = configs.copy()
            configs_1["data"]["dataset_paths"] = []
            dataset.get_dataloaders(configs_1)

        with self.assertRaises(AssertionError):
            configs_2 = configs.copy()
            configs_2["data"]["dataset_paths"] = "Target"
            dataset.get_dataloaders(configs_2)

        with self.assertRaises(AssertionError):
            configs_3 = configs.copy()
            del configs_3["data"]
            dataset.get_dataloaders(configs_3)

        with self.assertRaises(AssertionError):
            configs_4 = configs.copy()
            del configs_4["data"]["dataset_paths"]
            dataset.get_dataloaders(configs_4)


if __name__ == "__main__":
    unittest.main()