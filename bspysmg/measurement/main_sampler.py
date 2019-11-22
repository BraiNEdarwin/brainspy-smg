from bspyproc.processors.processor_mgr import get_hardware_processor
from bspyalgo.utils.io import load_configs
from bspysmg.measurement.sampler_mgr import get_sampler
from bspysmg.data.outputs. import preprocess_data
CONFIGS = load_configs('configs/sampling/sampling_configs_template.json')
processor = get_hardware_processor(CONFIGS["processor"])
sampler = get_sampler(CONFIGS)

RAW_DATA = sampler(CONFIGS)

DATA = preprocess_data(RAW_DATA)

## PLOTS? ##
