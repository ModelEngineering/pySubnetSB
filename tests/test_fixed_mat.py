from sirn.fixed_matrix import FixedMatrix  # type: ignore

import copy
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


#############################
# Tests
#############################
class TestFixedMat(unittest.TestCase):

    def setUp(self):
        self.arr = copy.copy(MAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        fixed_mat = FixedMatrix(self.arr)
        self.assertTrue(np.all(fixed_mat.arr == self.arr))

    def testMakeTrinaryMatrix(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            try:
                fixed_matrix = FixedMatrix.makeTrinaryMatrix(3, 3, prob0=0.9)
            except RuntimeError:
                continue
            matrix_sq = fixed_matrix.arr*fixed_matrix.arr
            is_nozerorow = np.all(matrix_sq.sum(axis=1) > 0)
            is_nozerocol = np.all(matrix_sq.sum(axis=0) > 0)
            self.assertTrue(is_nozerorow and is_nozerocol)
            break

        

if __name__ == '__main__':
    unittest.main()