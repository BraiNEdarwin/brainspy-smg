import os
import numpy as np
import matplotlib.pyplot as plt


def plot_error_hist(targets, prediction, error, mse, save_dir, name="error"):
    plt.figure()
    plt.title('Predicted vs True values')
    plt.subplot(1, 2, 1)
    plt.plot(targets, prediction, ".")
    plt.xlabel("True Output (nA)")
    plt.ylabel("Predicted Output (nA)")
    targets_and_prediction_array = np.concatenate((targets, prediction))
    min_out = np.min(targets_and_prediction_array)
    max_out = np.max(targets_and_prediction_array)
    plt.plot(np.linspace(min_out, max_out), np.linspace(min_out, max_out), "k")
    plt.title(f"RMSE {np.sqrt(mse)} (nA)")
    plt.subplot(1, 2, 2)
    plt.hist(np.reshape(error, error.size), 500)
    x_lim = 0.25 * np.max([np.abs(error.min()), error.max()])
    plt.xlim([-x_lim, x_lim])
    plt.title("Error histogram (nA) ")
    fig_loc = os.path.join(save_dir, name)
    plt.tight_layout()
    plt.savefig(fig_loc, dpi=300)
    plt.close()


def plot_error_vs_output(targets, error, save_dir, name="error_vs_output"):
    plt.figure()
    plt.plot(targets, error, ".")
    plt.plot(
        np.linspace(
            targets.min(),
            targets.max(),
            len(error),
        ),
        np.zeros_like(error),
    )
    plt.title("Error vs Output")
    plt.xlabel("Output (nA)")
    plt.ylabel("Error (nA)")
    fig_loc = os.path.join(save_dir, name)
    plt.savefig(fig_loc, dpi=300)
    plt.close()

