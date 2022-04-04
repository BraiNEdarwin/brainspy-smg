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
            self.fail("Failed in creating error hist")
        
        with self.assertRaises(AttributeError):
            plots.plot_error_hist(list(target), list(predictions), list(error), mse, ".")
        
        with self.assertRaises(TypeError):
            plots.plot_error_hist()
        
        with self.assertRaises(AssertionError):
            plots.plot_error_hist(target, predictions, np.array([0]), mse, ".")
        
        with self.assertRaises(AssertionError):
            plots.plot_error_hist(target, predictions, error, -1, ".")

    def test_error_output(self):

        target = np.random.uniform(0, 1, size=100)
        error = np.random.uniform(0, 1, size=100)

        try:
            plots.plot_error_vs_output(target, error, ".")
        except:
            self.fail("Failed in creating error vs output hist")
        
        with self.assertRaises(AttributeError):
            plots.plot_error_vs_output(list(target), list(error), ".")
        
        with self.assertRaises(TypeError):
            plots.plot_error_vs_output()
        
        with self.assertRaises(AssertionError):
            plots.plot_error_vs_output(target, np.array([0]), ".")

    def test_output_hist(self):

        output = np.random.uniform(0, 1, size=100)

        try:
            plots.output_hist(output, ".", 5, True)
            plots.output_hist(output, ".", 10, False)
            plots.output_hist(output, ".")
            plots.output_hist(list(output), ".")
        except:
            self.fail("Fail in creating output hist")

        with self.assertRaises(TypeError):
            plots.output_hist()

        with self.assertRaises(TypeError):
            plots.output_hist(output, ".", 10.1)

    def test_iv_plot(self):

        result = np.random.uniform(0, 1, size=100)

        try:
            plots.iv_plot(result, 5)
            plots.iv_plot(result, 10, ".", False)
            plots.iv_plot(result, 1.1, ".", True)
            plots.iv_plot(list(result), 2, ".", True, True)
        except:
            self.fail("Fail in creating output hist")
        
        with self.assertRaises(TypeError):
            plots.iv_plot()

        with self.assertRaises(FileNotFoundError):
            plots.iv_plot(result, 10.1, ",", True)


if __name__ == '__main__':
    unittest.main()
