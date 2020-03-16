
import matplotlib.pyplot as plt
from bspyalgo.utils.io import load_configs
from bspysmg.measurement.data.output.sampler_mgr import Sampler
from bspysmg.measurement.data.processing.postprocessing import post_process

CONFIGS = load_configs('configs/sampling/sampling_configs_template_cdaq_to_cdaq.json')
sampler = Sampler(CONFIGS)
path_to_data = sampler.get_data()

INPUTS, OUTPUTS, INFO_DICT = post_process(path_to_data, clipping_value=[-110, 110])

print(f"max out {OUTPUTS.max()} max min {OUTPUTS.min()} shape {OUTPUTS.shape}")
plt.show()
