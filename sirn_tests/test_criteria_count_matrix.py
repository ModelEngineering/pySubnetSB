from sirn.criteria_count_matrix import CriteriaCountMatrix  # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[0, 1], [1, 0], [0, 0]])
NROW, NCOL = MAT.shape


#############################
# Tests
#############################
class TestCriteriaMatrix(unittest.TestCase):

    def setUp(self):
        self.cc_matrix = CriteriaCountMatrix(MAT)

    def testMakeCriteriaCountMatrix(self):
        if IGNORE_TEST:
            return
        trues = [v == NCOL for v in np.sum(self.cc_matrix.values, axis=1)]
        self.assertTrue(all(trues))

    def testMakeCriteriaCountMatrixScale(self):
        if IGNORE_TEST:
            return
        for _ in range(1000):
            nrow, ncol = (100, 100)
            cc_matrix = CriteriaCountMatrix(np.random.randint(-10, 10, (nrow, ncol)))
            trues = [v == ncol for v in np.sum(cc_matrix.values, axis=1)]
            self.assertTrue(all(trues))



if __name__ == '__main__':
    unittest.main()