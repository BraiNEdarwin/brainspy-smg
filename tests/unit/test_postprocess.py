import unittest
from bspysmg.data import postprocess


class Test_PostProcess(unittest.TestCase):
    def test_get_sampling_data(self):
        with self.assertRaises(TypeError):
            inputs, outputs = postprocess.get_sampling_data()

        try:
            inputs, outputs = postprocess.get_sampling_data(
                "tests/data/IO.dat", 7, 1)
        except Exception:
            self.fail("Failed Execution: get_sampling_data")

        with self.assertRaises(AssertionError):
            inputs, outputs = postprocess.get_sampling_data(
                "tests/data/IO.dat", 3, 1)
        with self.assertRaises(OSError):
            inputs, outputs = postprocess.get_sampling_data(
                "tests/data/IOtemp.dat", 1, 5)

        with self.assertRaises(AssertionError):
            inputs, outputs = postprocess.get_sampling_data(
                "tests/data/IO.dat", 6.1, 1.9)
        with self.assertRaises(AssertionError):
            inputs, outputs = postprocess.get_sampling_data(
                "tests/data/IO.dat", 3.1, 1.2)

    def test_post_process(self):
        with self.assertRaises(TypeError):
            inputs, output, configs = postprocess.post_process()

        try:
            inputs, output, configs = postprocess.post_process("tests/data/")
            inputs, output, configs = postprocess.post_process(
                "tests/data/", [-4, 4])
        except Exception:
            self.fail("Failed Execution: post_process()")

        with self.assertRaises(FileNotFoundError):
            inputs, outputs = postprocess.post_process("not_existing_file")

        with self.assertRaises(TypeError):
            inputs, output, configs = postprocess.post_process(
                "tests/data/", [-4, 4], 41.5, 15.1)


if __name__ == "__main__":
    unittest.main()