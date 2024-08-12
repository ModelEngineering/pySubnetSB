from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[0, 1], [1, 0], [0, 0]])
MAT1 = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
NROW, NCOL = MAT.shape
CRITERIA_VECTOR = CriteriaVector([0.5])


#############################
# Tests
#############################
class TestPairCriteriaMatrix(unittest.TestCase):

    def setUp(self):
        self.pcc_mat = PairCriteriaCountMatrix(MAT1, criteria_vector=CRITERIA_VECTOR)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        repr = str(self.pcc_mat)
        self.assertTrue(isinstance(repr, str))
        self.assertTrue(np.all(self.pcc_mat.sorted_mat.values.shape == self.pcc_mat.values.shape))

    def testMakeCriteriaCountMatrixScale(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            nrow, ncol = (100, 100)
            pcc_mat = PairCriteriaCountMatrix(np.random.randint(-10, 10, (nrow, ncol)))
            for mat in pcc_mat.values:
                trues = [v == ncol for v in np.sum(mat, axis=1)]
                self.assertTrue(all(trues))

    def testIsEqual(self):
        if IGNORE_TEST:
            return
        result = self.pcc_mat.isEqual(self.pcc_mat, range(self.pcc_mat.num_row))
        self.assertTrue(result)
        #
        self.pcc_mat.values[0, 0] = 2
        result = self.pcc_mat.isEqual(self.pcc_mat, range(self.pcc_mat.num_row))
        self.assertTrue(result)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        copy_mat = self.pcc_mat.copy()
        self.assertTrue(self.pcc_mat == copy_mat)

    def testGetTargetArray(self):
        if IGNORE_TEST:
            return
        assignment = np.array(range(self.pcc_mat.num_row))
        reference = self.pcc_mat.getReferenceArray()
        target = self.pcc_mat.getTargetArray(assignment)
        self.assertTrue(np.all(reference == target))

    def testIsEqualScale(self):
        if IGNORE_TEST:
            return
        nrow, ncol = (10, 20)
        pcc_mat = PairCriteriaCountMatrix(np.random.randint(-10, 10, (nrow, ncol)))
        reference = pcc_mat.getReferenceArray()
        for _ in range(100000):
            assignment = np.random.permutation(range(pcc_mat.num_row))
            target = pcc_mat.getTargetArray(assignment)
            self.assertFalse(np.all(reference == target))


if __name__ == '__main__':
    unittest.main(failfast=True)