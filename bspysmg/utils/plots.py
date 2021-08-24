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


def plot_waves(self, inputs, outputs, input_no, output_no, batch, legend):
    plt.figure()
    plt.suptitle(f'Data for NN training in batch {batch}')
    plt.subplot(211)
    plt.plot(inputs)
    plt.ylabel('Inputs (V)')
    plt.xlabel('Time points (a.u.)')
    plt.legend(legend[:input_no])
    plt.subplot(212)
    plt.plot(outputs)
    plt.ylabel('Outputs (nA)')
    plt.legend(legend[-output_no:])
    plt.xlabel('Time points (a.u.)')
    plt.tight_layout()
    plt.savefig(os.path.join(self.configs["save_directory"], 'example_batch'))
    plt.close()


def output_hist(outputs, data_dir, bins=100, show=False):
    plt.figure()
    plt.title("Output Histogram")
    plt.hist(outputs, bins=bins)
    plt.ylabel("Counts")
    plt.xlabel("Raw output (nA)")
    if show:
        plt.show()
    plt.savefig(data_dir + "/output_distribution")
    plt.close()


# def plot(self, x, y):
#     for i in range(np.shape(y)[1]):
#         plt.figure()
#         plt.plot(x)
#         plt.plot(y)
#         plt.show()


def iv_plot(configs, inputs, output):
    ylabeldist = -10
    electrode_id = 0
    cmap = plt.get_cmap("tab10")
    for k, dev in enumerate(configs['devices']):
        fig, axs = plt.subplots(2, 4)
        # plt.grid(True)
        fig.suptitle('Device ' + dev + ' - Input voltage vs Output current')
        for i in range(2):
            for j in range(4):
                exp_index = j + i * 4
                exp = "IV" + str(exp_index + 1)
                if exp_index < 7:
                    if configs["driver"]['instruments_setup'][dev][
                            "activation_channel_mask"][exp_index] == 1:
                        masked_idx = sum(
                            configs["driver"]['instruments_setup'][dev]
                            ["activation_channel_mask"][:exp_index + 1]) - 1
                        axs[i, j].plot(inputs[exp][dev][masked_idx],
                                       output[exp][dev],
                                       color=cmap(exp_index))
                        axs[i, j].set_ylabel('output (nA)',
                                             labelpad=ylabeldist)
                        axs[i, j].set_xlabel('input (V)', labelpad=1)
                        axs[i, j].xaxis.grid(True)
                        axs[i, j].yaxis.grid(True)
                    else:
                        # if self.configs["driver"]['instruments_setup'][
                        #         dev]["activation_channel_mask"][
                        #             exp_index] == 1:
                        #     axs[i,
                        #         j].plot(input_waveform[exp_index]
                        #                 [:, electrode_id])

                        #     axs[i, j].set_title(
                        #         devlist[dev]["activation_channels"]
                        #         [exp_index])
                        axs[i, j].plot([])
                        axs[i, j].set_xlabel('Channel Masked')
                    electrode_id += 1
                else:
                    for z, key in enumerate(inputs.keys()):
                        m = 0
                        if configs["driver"]['instruments_setup'][dev][
                                "activation_channel_mask"][z] == 1:
                            masked_idx = sum(
                                configs["driver"]['instruments_setup'][dev]
                                ["activation_channel_mask"][:z + 1]) - 1
                            axs[i, j].plot(inputs[key][dev][masked_idx],
                                           label="IV" + str(z),
                                           color=cmap(z))
                            m += 1
                    #axs[i, j].yaxis.tick_right()
                    #axs[i, j].yaxis.set_label_position("right")
                    axs[i, j].set_ylabel('input (V)')
                    axs[i, j].set_xlabel('points', labelpad=1)
                    axs[i, j].set_title("Input input_signal")
                    axs[i, j].xaxis.grid(True)
                    axs[i, j].yaxis.grid(True)
                    axs[i, j].legend()
    plt.subplots_adjust(hspace=0.3, wspace=0.35)
    plt.show()
