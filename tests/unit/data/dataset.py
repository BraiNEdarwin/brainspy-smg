import unittest
from bspysmg.data import dataset
from brainspy.utils.io import load_configs

class Test_Dataset(unittest.TestCase):

    def test_empty_inputs(self):
        with self.assertRaises(TypeError):
            datasetClass = dataset.ModelDataset()
    
    def test_normal_inputs(self):
        try:
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=10, tag="train")
        except:
            self.fail("Failed Execution: Steps - 10, Tag - train")
        try:
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=5, tag="validation")
        except:
            self.fail("Failed Execution: Steps - 5, Tag - validation")
        try:
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=1, tag="test")
        except:
            self.fail("Failed Execution: Steps - 1, Tag - test")

    def test_check_file_exists(self):
        with self.assertRaises(FileNotFoundError):
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\processed_data.npz")
    
    def test_steps_values(self):
        with self.assertRaises(TypeError):
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=0.1, tag="train")
        with self.assertRaises(TypeError):
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps="10", tag="train")
        with self.assertRaises(TypeError):
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=[1, 2, 3], tag="train")
    
    def test_tag_values(self):

        with self.assertRaises(ValueError):
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=1, tag="target")
        with self.assertRaises(ValueError):
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=1, tag=[1, 2, 3])
        with self.assertRaises(ValueError):
            datasetClass = dataset.ModelDataset("C:\\Users\\sriku\\Downloads\\postprocessed_data.npz", steps=1, tag=6.5)
    
    def test_get_dataloaders(self):
        configs = load_configs("configs/training/smg_configs_template.yaml")
        configs["data"]["dataset_paths"] = ["C:\\Users\\sriku\\Downloads\\postprocessed_data.npz"]*3
        try:
            dataset.get_dataloaders(configs)
        except:
            self.fail("Execution Failed: configs_path - configs/training/smg_configs_template.yaml")
        
        with self.assertRaises(AssertionError):
            configs_1 = configs.copy()
            configs_1["data"]["dataset_paths"] = []
            dataset.get_dataloaders(configs_1)

        with self.assertRaises(AssertionError):
            configs_2 = configs.copy()
            configs_2["data"]["dataset_paths"] = "Target"
            dataset.get_dataloaders(configs_2)
        
if __name__ == "__main__":
    unittest.main()