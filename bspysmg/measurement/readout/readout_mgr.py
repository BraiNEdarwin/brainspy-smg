import numpy as np
import time
from bspysmg.measurement.data.inputs.input_mgr import ramped_input_generator
from bspyproc import processor


def wave_sampler(configs):
    # Initialize output containers
    data = np.zeros((int(configs['sample_time'] * configs['sample_frequency']), 1))

    batches = int(configs['sample_frequency'] * configs['sample_time'] / configs['sample_points'])
    ramp = int(configs['ramp_time'] * configs['sample_frequency'])

    for i in range(batches):
        start_batch = time.time()

        time_points = np.arange(i * configs['sample_points'], (i + 1) * configs['sample_points'])
        waves_ramped = ramped_input_generator(time_points, configs)
        # dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(waves_ramped, configs['sample_frequency'])
        dataRamped = processor.get_output(waves_ramped)
        data[i * configs['sample_points']: (i + 1) * configs['sample_points'], 0] = dataRamped[0, ramp: ramp + time_points]

        if i % 10 == 0:  # Save after every 10 batches
            print('Saving...')
            # TODO: implement saving
            SaveLib.saveExperiment(configs['config_src'], saveirectory,
                                   output=data * configs['amplification'] / configs['postgain'],
                                   freq=configs['freq'],
                                   sampleTime=configs['sample_time'],
                                   sample_frequency=configs['sample_frequency'],
                                   phase=configs['phase'],
                                   amplitude=configs['amplitude'],
                                   offset=configs['offset'],
                                   amplification=configs['amplification'],
                                   electrodeSetup=configs['electrode_setup'],
                                   gain_info=configs['gain_info'],
                                   filename='training_nn_data')
        end_batch = time.time()
        print('Data collection for part ' + str(i + 1) + ' of ' + str(batches) + ' took ' + str(end_batch - start_batch) + ' sec.')
