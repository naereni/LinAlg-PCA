import unittest
import sys

from test_matrix_basic import TestMatrixBasic
from test_matrix_arithmetic import TestMatrixArithmetic
from test_matrix_linalg import TestMatrixLinAlg
from test_utils import TestUtils


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTest(loader.loadTestsFromTestCase(TestMatrixBasic))
    suite.addTest(loader.loadTestsFromTestCase(TestMatrixArithmetic))
    suite.addTest(loader.loadTestsFromTestCase(TestMatrixLinAlg))
    suite.addTest(loader.loadTestsFromTestCase(TestUtils))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
