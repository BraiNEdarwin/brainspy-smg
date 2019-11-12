def wave_sampler(configs):
    # Initialize output containers
    data = np.zeros((int(configs['sample_time'] * configs['sample_frequency']), 1))

    nr_batches = int(configs['sample_frequency'] * configs['sample_time'] / configs['sample_points'])

    for i in range(batches):
        start_batch = time.time()

        time_points = np.arange(i * configs['sample_points'], (i + 1) * configs['sample_points'])
        waves = input_generator(time_points, configs)
        # Use 0.5 second to ramp up to the value where data aqcuisition stopped previous iteration
        # and 0.5 second to ramp down after the batch is done
        waves_ramped = np.zeros((waves.shape[0], waves.shape[1] + int(configs['sample_frequency'])))
        dataRamped = np.zeros(waves_ramped.shape[1])
        for j in range(waves_ramped.shape[0]):
            waves_ramped[j, 0:int(0.5 * configs['sample_frequency'])] = np.linspace(0, waves[j, 0], int(0.5 * configs['sample_frequency']))
            waves_ramped[j, int(0.5 * configs['sample_frequency']): int(0.5 * configs['sample_frequency']) + waves.shape[1]] = waves[j, :]
            waves_ramped[j, int(0.5 * configs['sample_frequency']) + waves.shape[1]:] = np.linspace(waves[j, -1], 0, int(0.5 * configs['sample_frequency']))

        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(waves_ramped, configs['sample_frequency'])
        data[i * configs['sample_points']: (i + 1) * configs['sample_points'], 0] = dataRamped[0, int(0.5 * configs['sample_frequency']):int(0.5 * configs['sample_frequency']) + waves.shape[1]]

    if i % 10 == 0:  # Save after every 10 batches
        print('Saving...')
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
