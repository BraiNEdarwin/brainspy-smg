import unittest
from bspysmg.utils import plots
import numpy as np
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs

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
            plots.plot_error_hist(list(target), list(predictions), list(error),
                                  mse, ".")

        with self.assertRaises(TypeError):
            plots.plot_error_hist()

        with self.assertRaises(AssertionError):
            plots.plot_error_hist(target, predictions, np.array([0]), mse, ".")

        with self.assertRaises(AssertionError):
            plots.plot_error_hist(target, predictions, error, -1, ".")
        plt.close("all")

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
        plt.close("all")

    def test_output_hist(self):

        output = np.random.uniform(0, 1, size=100)

        try:
            plots.output_hist(output, ".", 5)
            plots.output_hist(output, ".", 10, False)
            plots.output_hist(output, ".")
            plots.output_hist(list(output), ".")
        except:
            self.fail("Fail in creating output hist")

        with self.assertRaises(TypeError):
            plots.output_hist()

        with self.assertRaises(TypeError):
            plots.output_hist(output, ".", 10.1)
        plt.close("all")

    def test_iv_plot(self):

        inputs = np.random.randn(100, 7)
        result = np.random.randn(100, 1)

        try:
            plots.iv_plot(result=result, inputs=inputs, input_electrode=1)
            #plots.iv_plot(inputs, result, ".", True)
            #plots.iv_plot(list(result), 2, ".", True, True)
        except Exception:
            self.fail("Fail in creating IV curve.")

        with self.assertRaises(TypeError):
            plots.iv_plot()

        with self.assertRaises(TypeError):
            plots.iv_plot(result, 5)

        with self.assertRaises(ValueError):
            plots.iv_plot(result, 10.1, ",")
        plt.close("all")

    def test_plot_waves(self):

        inp = np.random.uniform(0, 1, size=100)
        out = np.random.uniform(0, 1, size=100)
        legend = np.array(["Training", "Training"])

        try:
            plots.plot_waves(inp, out, 0, 1, 1, legend, "./")
            plots.plot_waves(inp, out, 0, 1, 1.1, legend, "./")
            plots.plot_waves(list(inp), list(out), 0, 1, 1.1, legend, "./")
        except:
            self.fail("Fail in creating input and output waves")

        with self.assertRaises(TypeError):
            plots.plot_waves()
        plt.close("all")

    def test_multiple_plots(self):
        configs = load_configs(
            'configs/utils/brains_ivcurve_template.yaml')
        inputs = load_configs('tests/data/ins.yaml')
        output = load_configs('tests/data/outs.yaml')
        configs['driver']['instruments_setup']['activation_sampling_frequency'] = 100
        configs['driver']['instruments_setup']['readout_sampling_frequency'] = 100
        configs['driver']['instruments_setup']['average_io_point_difference'] = True
        plots.multi_iv_plot(configs, inputs, output, show_plot=False)
        configs['driver']['instruments_setup']['average_io_point_difference'] = False
        plots.multi_iv_plot(configs, inputs, output, show_plot=False)


if __name__ == '__main__':
    unittest.main()
