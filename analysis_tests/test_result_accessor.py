from analysis.result_accessor import ResultAccessor # type: ignore
import sirn.constants as cnn  # type: ignore
import analysis.constants as cn  # type: ignore

import os
import copy
import numpy as np  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
DIRECTORY = "Oscillators_DOE_JUNE_10_17565"
IS_STRONG = False
MAX_NUM_PERM = 100


#############################
# Tests
#############################
class TestResultAccessor(unittest.TestCase):

    def setUp(self):
        self.accessor = ResultAccessor(DIRECTORY, is_strong=IS_STRONG, max_num_perm=MAX_NUM_PERM,
                                       data_dir=cn.TEST_DIR)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        for is_strong in [True, False]:
            for directory in cnn.OSCILLATOR_DIRS:
                accessor = ResultAccessor(directory, is_strong=is_strong)
                assert(isinstance(accessor.results, list))
                assert(isinstance(accessor.results[0], str))

        

if __name__ == '__main__':
    unittest.main()