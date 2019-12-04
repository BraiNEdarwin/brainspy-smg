
from bspyalgo.utils.io import load_configs
from bspysmg.measurement.data.output.sampler_mgr import Sampler
from bspysmg.measurement.data.processing.postprocessing import post_process

CONFIGS = load_configs('configs/sampling/toy_sampling_configs_template.json')
sampler = Sampler(CONFIGS)
path_to_data = sampler.get_data()

INPUTS, OUTPUTS, INFO_DICT = post_process(path_to_data)
