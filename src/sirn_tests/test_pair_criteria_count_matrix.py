from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[0, 1], [1, 0], [0, 0]])
MAT1 = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
MAT2 = np.array([[-1, 0], [1, 2]])
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
        self.checkPairCriteriaCountMatrix(self.pcc_mat)

    def testMakePairCriteriaCountMatrix(self):
        if IGNORE_TEST:
            return
        pcc_mat = PairCriteriaCountMatrix(MAT2, criteria_vector=CRITERIA_VECTOR)
        self.checkPairCriteriaCountMatrix(pcc_mat)

    def checkPairCriteriaCountMatrix(self, pcc_mat):
        nmat, nrow, _ = pcc_mat.values.shape
        ncol = pcc_mat.array.shape[1]
        for imat in range(nmat):
            for irow in range(nrow):
                self.assertEqual(ncol, np.sum(pcc_mat.values[imat, irow, :]))
        self.assertEqual(len(pcc_mat.criteria_vector_idx_pairs), pcc_mat.num_column)

    def testMakeCriteriaCountMatrixScale(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            nrow, ncol = (100, 100)
            pcc_mat = PairCriteriaCountMatrix(np.random.randint(-10, 10, (nrow, ncol)))
            self.checkPairCriteriaCountMatrix(pcc_mat)
            for mat in pcc_mat.values:
                trues = [v == ncol for v in np.sum(mat, axis=1)]
                self.assertTrue(all(trues))

    def testIsEqual(self):
        if IGNORE_TEST:
            return
        assignment = np.array(range(self.pcc_mat.num_row))
        assignment = np.reshape(assignment, (1, len(assignment)))
        result = self.pcc_mat.isEqual(self.pcc_mat, assignment)
        self.assertTrue(result)
        #
        self.pcc_mat.values[0, 0] = 2
        result = self.pcc_mat.isEqual(self.pcc_mat, assignment)
        self.assertTrue(result)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        copy_mat = self.pcc_mat.copy()
        self.assertTrue(self.pcc_mat == copy_mat)

    def testGetTargetArray(self):
        if IGNORE_TEST:
            return
        assignment = [np.array(range(self.pcc_mat.num_row))]
        assignment = np.reshape(assignment, (1, self.pcc_mat.num_row))
        reference = self.pcc_mat.getReferenceArray()
        target = self.pcc_mat.getTargetArray(assignment)
        self.assertTrue(np.all(reference == target))
        for idx in assignment[0, :-1]:
            self.assertTrue(np.all(reference[idx, :] == self.pcc_mat.values[idx, idx+1, :]))

    def testGetTargetArrayScale(self):
        if IGNORE_TEST:
            return
        size = 20
        for _ in range(100):
            pcc_mat = PairCriteriaCountMatrix(np.random.randint(-2, 3, (size, size)))
            assignment = np.array(range(pcc_mat.num_row))
            assignment = np.reshape(assignment, (1, len(assignment)))
            reference = pcc_mat.getReferenceArray()
            target = pcc_mat.getTargetArray(assignment)
            self.assertTrue(np.all(reference == target))
            for idx in assignment[:-1]:
                self.assertTrue(np.all(reference[idx, :] == pcc_mat.values[idx, idx+1, :]))

    def testIsEqualScale(self):
        if IGNORE_TEST:
            return
        nrow, ncol = (10, 10)
        pcc_mat = PairCriteriaCountMatrix(np.random.randint(-2, 3, (nrow, ncol)))
        reference = pcc_mat.getReferenceArray()
        num_iteration = 10000
        num_match = 0
        for _ in range(num_iteration):
            assignment = np.random.permutation(range(pcc_mat.num_row))
            assignment = np.reshape(assignment, (1, len(assignment)))
            target = pcc_mat.getTargetArray(assignment)
            num_match += np.all(reference == target)
        self.assertLess(num_match/num_iteration, 0.01)


if __name__ == '__main__':
    unittest.main(failfast=True)