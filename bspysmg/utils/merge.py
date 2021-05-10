import numpy as np


def merge_postprocessed_data(file_names, output_file_name='merged_postprocessed_data.npz'):
    ref_data = dict(np.load(file_names[0], allow_pickle='True'))
    for i in range(1, len(file_names)):
        data = np.load(file_names[i])
        for key in list(data):
            if key != 'info':
                ref_data[key] = np.append(ref_data[key], data[key], axis=0) 
    np.savez(output_file_name, **ref_data)


if __name__ == "__main__":
    file_names = ['tmp/data/training/Brains_testing_2020_09_04_182557/postprocessed_data.npz', 'tmp/data/training/Brains_testing_2020_09_11_093200/postprocessed_data.npz']
    merge_postprocessed_data(file_names)
