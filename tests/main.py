"""
Module for running all tests on brainspy.
"""

import os
import unittest
import bspysmg

if __name__ == "__main__":
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')

    timestamp = datetime.today().strftime('%d-%m-%Y-%H:%M:%S')

    bspysmg.TEST_MODE = 'SIMULATION_PC'

    modules_to_test = unittest.defaultTestLoader.discover(start_dir="tests/",
                                                          pattern="*.py",
                                                          top_level_dir=None)
    
    runner = unittest.TextTestRunner()
    runner.run(modules_to_test)
    print(bspysmg.TEST_MODE)
