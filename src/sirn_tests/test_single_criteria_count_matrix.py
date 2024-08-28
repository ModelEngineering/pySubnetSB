from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore
from sirn.network import Network  # type: ignore

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
        self.assertTrue(isinstance("int" in str(type(self.scc_mat.row_order_independent_hash)), int))

    def testRowIndependentHash(self):
        if IGNORE_TEST:
            return
        def test(size=5, num_iteration=100):
            for _ in range(num_iteration):
                mat1 = np.random.randint(-2, 3, (size, 2*size))
                row_perm = np.random.permutation(size)
                scc_mat1 = SingleCriteriaCountMatrix(mat1)
                mat2 = mat1.copy()
                row_perm = np.random.permutation(size)
                mat2 = mat2[row_perm, :]
                scc_mat2 = SingleCriteriaCountMatrix(mat2)
                self.assertTrue(scc_mat1.row_order_independent_hash == scc_mat2.row_order_independent_hash)
        #
        test(5, 100)

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