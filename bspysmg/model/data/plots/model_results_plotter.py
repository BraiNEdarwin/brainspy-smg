import os
import numpy as np
import matplotlib.pyplot as plt


def plot_error_hist(targets, prediction, error, mse, save_dir, name='test_error'):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(targets, prediction, '.')
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    targets_and_prediction_array = np.concatenate((targets, prediction))
    min_out = np.min(targets_and_prediction_array)
    max_out = np.max(targets_and_prediction_array)
    plt.plot(np.linspace(min_out, max_out), np.linspace(min_out, max_out), 'k')
    plt.title(f'Predicted vs True values:\n MSE {mse}')
    plt.subplot(1, 2, 2)
    plt.hist(np.reshape(error, error.size), 500)
    x_lim = 0.25 * np.max([np.abs(error.min()), error.max()])
    plt.xlim([-x_lim, x_lim])
    plt.title('Scaled error histogram')
    fig_loc = os.path.join(save_dir, name)
    plt.savefig(fig_loc, dpi=300)
    plt.close()


def plot_error_vs_output(targets, error, save_dir, name='test_error_vs_output'):
    plt.figure()
    plt.plot(targets, error, '.')
    plt.plot(np.linspace(targets.min(), targets.max(), len(error)), np.zeros_like(error))
    plt.title('Error vs Output')
    plt.xlabel('Output')
    plt.ylabel('Error')
    fig_loc = os.path.join(save_dir, name)
    plt.savefig(fig_loc, dpi=300)
    plt.close()


def plot_all(targets, outputs, results_dir, name=''):
    error = outputs - targets
    mse = np.mean(error ** 2)
    print(f'Model MSE on {name}: {mse}')
    plot_error_vs_output(targets, error, results_dir, name=name + '_test_error_vs_output')
    plot_error_hist(targets, outputs, error, mse, results_dir, name=name + '_test_error')
    return mse
