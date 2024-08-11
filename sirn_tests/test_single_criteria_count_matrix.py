from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore

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
class TestSingleCriteriaMatrix(unittest.TestCase):

    def setUp(self):
        self.scc_mat = SingleCriteriaCountMatrix(MAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        repr = str(self.scc_mat)
        self.assertTrue(isinstance(repr, str))
        self.assertTrue(np.all(self.scc_mat.row_hashes == [10100, 10100,    200]))

    def testMakeCriteriaCountMatrix(self):
        if IGNORE_TEST:
            return
        trues = [v == NCOL for v in np.sum(self.scc_mat.values, axis=1)]
        self.assertTrue(all(trues))

    def testMakeCriteriaCountMatrixScale(self):
        if IGNORE_TEST:
            return
        for _ in range(1000):
            nrow, ncol = (100, 100)
            scc_mat = SingleCriteriaCountMatrix(np.random.randint(-10, 10, (nrow, ncol)))
            trues = [v == ncol for v in np.sum(scc_mat.values, axis=1)]
            self.assertTrue(all(trues))

    def testIsEqual(self):
        if IGNORE_TEST:
            return
        result = self.scc_mat.isEqual(self.scc_mat, range(self.scc_mat.num_row))
        self.assertTrue(result)
        #
        self.scc_mat.values[0, 0] = 2
        result = self.scc_mat.isEqual(self.scc_mat, range(self.scc_mat.num_row))
        self.assertTrue(result)
    
    def testIsEqualScale(self):
        if IGNORE_TEST:
            return
        nrow, ncol = (10, 20)
        scc_mat = SingleCriteriaCountMatrix(np.random.randint(-10, 10, (nrow, ncol)))
        for _ in range(100000):
            self.assertTrue(scc_mat.isEqual(scc_mat, range(scc_mat.num_row)))
            self.assertTrue(scc_mat.isLessEqual(scc_mat, range(scc_mat.num_row)))

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        copy_mat = self.scc_mat.copy()
        self.assertTrue(self.scc_mat == copy_mat)


if __name__ == '__main__':
    unittest.main(failfast=True)