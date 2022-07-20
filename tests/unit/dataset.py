import unittest
from bspysmg.data import dataset
from brainspy.utils.io import load_configs


class Test_Dataset(unittest.TestCase):
    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            dataset.ModelDataset()

    def test_normal_inputs(self):
        try:
            dataset.ModelDataset("tests/data/testing_postprocessed_data.npz",
                                 steps=10)
        except Exception:
            self.fail("Failed Execution: Steps - 10, Tag - train")
        try:
            dataset.ModelDataset("tests/data/testing_postprocessed_data.npz",
                                 steps=5)
        except Exception:
            self.fail("Failed Execution: Steps - 5, Tag - validation")
        try:
            dataset.ModelDataset("tests/data/testing_postprocessed_data.npz",
                                 steps=1)
        except Exception:
            self.fail("Failed Execution: Steps - 1, Tag - test")

    def test_check_file_exists(self):
        with self.assertRaises(FileNotFoundError):
            dataset.ModelDataset("test_fail.npz")

    def test_steps_values(self):
        with self.assertRaises(TypeError):
            dataset.ModelDataset("tests/data/testing_postprocessed_data.npz",
                                 steps=0.1)
        with self.assertRaises(TypeError):
            dataset.ModelDataset("tests/data/testing_postprocessed_data.npz",
                                 steps="10")
        with self.assertRaises(TypeError):
            dataset.ModelDataset("tests/data/testing_postprocessed_data.npz",
                                 steps=[1, 2, 3])

    def test_get_dataloaders(self):
        configs = load_configs("configs/training/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = [
            "tests/data/testing_postprocessed_data.npz"
        ] * 3
        try:
            dataset.get_dataloaders(configs)
        except Exception:
            self.fail(
                "Execution Failed: configs_path - configs/training/smg_configs_template.yaml"
            )

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