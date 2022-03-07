import unittest
from bspysmg.utils import plots
import numpy as np


class Test_Plots(unittest.TestCase):

    def test_error_hist(self):

        target = np.random.uniform(0, 1, size=100)
        predictions = np.random.uniform(0, 1, size=100)
        error = target - predictions
        mse = np.mean(error**2)

        try:
            plots.plot_error_hist(target, predictions, error, mse, ".")
        except:
            self.fail("Fail in creating error hist")
        
        with self.assertRaises(AttributeError):
            plots.plot_error_hist(list(target), list(predictions), list(error), mse, ".")
        
        with self.assertRaises(TypeError):
            plots.plot_error_hist()
        

if __name__ == '__main__':
    unittest.main()
