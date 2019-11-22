
from bspyalgo.utils.io import load_configs
from bspysmg.measurement.data.output.sampler_mgr import get_sampler
# from bspysmg.measurement.data.processing import preprocess_data

CONFIGS = load_configs('configs/sampling/sampling_configs_template.json')
# TODO: use NN model as processor for debugging
sampler = get_sampler(CONFIGS)
RAW_DATA = sampler(CONFIGS)

# DATA = preprocess_data(RAW_DATA)

## PLOTS? ##
