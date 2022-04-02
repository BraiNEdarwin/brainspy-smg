import unittest
from bspysmg.model import training

class Test_PostProcess(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super(Test_PostProcess, self).__init__(*args, **kwargs)
        self.configs = {'results_base_dir': '.'}
        self.configs['model_structure'] = {
            'hidden_sizes': [90]*5,
            'D_in': 7,
            'D_out': 1,
            'activation': 'relu'
        }
        self.configs['hyperparameters'] = {
            'epochs': 5,
            'learning_rate': 1.0e-05
        }
        self.configs['data'] = {
            'dataset_paths': ["./postprocessed_data_train.npz",
                "./postprocessed_data_val.npz",
                "./postprocessed_data_test.npz"],
            'steps': 3,
            'batch_size': 128,
            'worker_no': 0,
            'pin_memory': False,
            'split_percentages': [1, 0]
        }

    def test_init_seed(self):
        with self.assertRaises(TypeError):
            training.init_seed()

        try:
            training.init_seed(self.configs)
        except:
            self.fail("Failed Execution: init_seed()")
        
        with self.assertRaises(TypeError):
            training.init_seed([])
    
    def test_surrogate_model(self):
        with self.assertRaises(TypeError):
            training.generate_surrogate_model()
        
        training.generate_surrogate_model(self.configs, main_folder='.')

if __name__ == "__main__":
    unittest.main()