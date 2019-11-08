
from SkyNEt.experiments.wave_search import transient_test
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
