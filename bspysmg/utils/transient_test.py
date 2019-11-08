import time
import numpy as np


def transient_test(waves, data, fs, sampleTime, n):
    T = 3  # Amount of time of sampling one datapoint
    rampT = int(fs / 2)
    iterations = 1  # iterate multiple times over the test cases
    testdata = np.zeros((iterations, n, (T * fs - 2 * rampT)))
    # testdata = np.zeros((n, T*fs))
    test_cases = np.random.randint(waves.shape[1], size=(1, n))  # Index for the wave
    difference = np.zeros((iterations, n, 1))

    for it in range(iterations):
        for i in range(n):
            start_wave = time.time()

            waves_ramped = np.zeros((waves.shape[0], T * fs))  # 1.5 second to ramp up to desired input, T-3 seconds measuring, 1.5 second to ramp input down to zero
            dataRamped = np.zeros(waves_ramped.shape[1])
            for j in range(waves_ramped.shape[0]):
                # Ramp up linearly (starting from value CV/2 since at CV=0V nothing happens anyway) and ramp back down to 0
                waves_ramped[j, 0:rampT] = np.linspace(0, waves[j, test_cases[0, i]], rampT)
                waves_ramped[j, rampT: rampT + (T * fs - 2 * rampT)] = np.ones((T * fs - 2 * rampT)) * waves[j, test_cases[0, i], np.newaxis]
                waves_ramped[j, rampT + (T * fs - 2 * rampT):] = np.linspace(waves[j, test_cases[0, i]], 0, rampT)

            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(waves_ramped, fs)
            testdata[it, i, :] = dataRamped[0, rampT: rampT + (T * fs - 2 * rampT)]
            # testdata[i,:] = dataRamped[0, :]

            difference[it, i, 0] = np.mean(testdata[it, i, :]) - data[test_cases[0, i]]
            end_wave = time.time()
            print('Transient test data point ' + str(it * n + i + 1) + ' of ' + str(iterations * n) + ' took ' + str(end_wave - start_wave) + ' sec.')

    return testdata, difference, test_cases


print("Testing for transients...")
print("Only for the last loaded data the transients are tested ")

ytestdata, difference, xtestdata = transient_test.transient_test(waves, data[(batches - 1) * configs['samplePoints']:(batches) * configs['samplePoints'], configs['sample_frequency'], configs['sampleTime'], configs['n'])
SaveLib.saveExperiment(configs['configSrc'], saveDirectory,
                        xtestdata=xtestdata,
                        ytestdata=ytestdata * configs['amplification'] / configs['postgain'],
                        diff=difference * configs['amplification'] / configs['postgain'],
                        output=data * configs['amplification'] / configs['postgain'],
                        freq=configs['freq'],
                        sampleTime=configs['sampleTime'],
                        sample_frequency=configs['sample_frequency'],
                        phase=configs['phase'],
                        amplitude=configs['amplitude'],
                        offset=configs['offset'],
                        amplification=configs['amplification'],
                        electrodeSetup=configs['electrodeSetup'],
                        gain_info=configs['gain_info'],
                        filename='training_NN_data')
