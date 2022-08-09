import os
import math
import torch

DATA_NAMES = ['training', 'validation', 'test']


def get_random_phases(activation_electrode_no=7, first_zero=True):
    phases = (torch.rand((3, activation_electrode_no)) -
              0.5) * 720  # Get values between -360 and 360 (degrees)
    phases *= (math.pi / 180)  # convert to radians
    if first_zero:  # Enable the training dataset to have zero phase
        phases[0] *= 0
    return phases.detach().cpu().numpy().tolist()

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from brainspy.utils.io import load_configs
    from bspysmg.data.sampling import Sampler
    from bspysmg.data.postprocess import post_process
    from bspysmg.model.training import generate_surrogate_model

    dataset_paths = []
    data_name_base = 'custom_model'
    number_batches = [3880,388,388]
    phases = get_random_phases()


    smg_configs = load_configs("configs/training/smg_configs_template.yaml")

    for i in range(len(DATA_NAMES)):
        sampler_configs = load_configs(
        'configs/sampling/sampling_configs_template_cdaq_to_cdaq.yaml')
        sampler_configs['data_name'] = data_name_base + DATA_NAMES[i]
        sampler_configs['input_data']['phase'] = phases[i]
        sampler_configs['input_data']['number_batches'] = number_batches[i]

        sampler = Sampler(sampler_configs)
        dataset_paths.append(sampler.sample())

        _, outputs, _ = post_process(dataset_paths[-1],
                                                clipping_value=None)
        dataset_paths[-1] = os.path.join(dataset_paths[-1],'postprocessed_data.npz')
        print(
            f"max out {outputs.max()} max min {outputs.min()} shape {outputs.shape}"
        )
    smg_configs['data']['dataset_paths'] = dataset_paths
    generate_surrogate_model(smg_configs)   
