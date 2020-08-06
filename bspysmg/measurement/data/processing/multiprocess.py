import os
import numpy as np


def process_multiple(main_dir):
    ELECTRODE_NO = 7
    shape = 0
    dirs = list([name for name in os.listdir(main_dir) if os.path.isdir(
        os.path.join(main_dir, name)) and not name.startswith('.')])

    assert len(dirs) > 0
    for i in range(len(dirs)):
        shape += np.load(os.path.join(main_dir, dirs[i], 'postprocessed_data.npz'), allow_pickle=True)['inputs'].shape[0]

    input_results = np.zeros([shape, ELECTRODE_NO])
    output_results = np.zeros([shape, 1])
    previous_shape = 0
    for i in range(len(dirs)):
        data = np.load(os.path.join(main_dir, dirs[i], 'postprocessed_data.npz'), allow_pickle=True)
        current_shape = previous_shape + data['inputs'].shape[0]
        input_results[previous_shape:current_shape] = data['inputs']
        output_results[previous_shape:current_shape] = data['outputs']
        previous_shape = current_shape
        info = data['info']

    info = dict(np.ndenumerate(info))[()]
    info['input_data']['input_distribution'] = 'mixed'
    info['input_data']['phase'] = 'mixed'
    index = np.random.permutation(np.arange(shape))
    input_results = input_results[index]
    output_results = output_results[index]

    limit = int(shape * 0.75)

    np.savez(os.path.join(main_dir, 'training_data'), inputs=input_results[:limit], outputs=output_results[:limit], info=info)
    np.savez(os.path.join(main_dir, 'test_data'), inputs=input_results[limit:], outputs=output_results[limit:], info=info)


if __name__ == '__main__':
    main_dir = "tmp/output/model_nips"
    # The post_process function should have a clipping value which is in an amplified scale.
    # E.g., for an amplitude of 100 -> 345.5
    process_multiple(main_dir)
