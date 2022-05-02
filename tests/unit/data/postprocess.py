import unittest
from bspysmg.data import postprocess

class Test_PostProcess(unittest.TestCase):

    def test_get_sampling_data(self):
        with self.assertRaises(TypeError):
            inputs, outputs = postprocess.get_sampling_data()
        
        try:
            inputs, outputs = postprocess.get_sampling_data("./IO.dat", 1, 5)
        except:
            self.fail("Failed Execution: get_sampling_data")
        
        with self.assertRaises(FileNotFoundError):
            inputs, outputs = postprocess.get_sampling_data("./IOtemp.dat", 1, 5)
        
        with self.assertRaises(TypeError):
            inputs, outputs = postprocess.get_sampling_data("./IO.dat", 1.1, 5.5)
    
    def test_post_process(self):
        with self.assertRaises(TypeError):
            inputs, output, configs = postprocess.post_process()
        
        try:
            inputs, output, configs = postprocess.post_process(".")
            inputs, output, configs = postprocess.post_process(".", [-4, 4])
        except:
            self.fail("Failed Execution: post_process()")
        
        with self.assertRaises(FileNotFoundError):
            inputs, outputs = postprocess.post_process("./downloads")
        
        with self.assertRaises(TypeError):
            inputs, output, configs = postprocess.post_process(".", [-4, 4], 41.5, 15.1)


if __name__ == "__main__":
    unittest.main()