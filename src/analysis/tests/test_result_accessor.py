from analysis.result_accessor import ResultAccessor # type: ignore
import sirn.constants as cnn

import copy
import numpy as np  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False


#############################
# Tests
#############################
class TestResultAccessor(unittest.TestCase):

    def setUp(self):
        self.accessor = ResultAccessor(cnn.OSCILLATOR_DIRS[0], is_strong=True)

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