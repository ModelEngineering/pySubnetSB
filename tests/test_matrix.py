from sirn.matrix import Matrix # type: ignore

import copy
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


#############################
# Tests
#############################
class TestMatrixClassifier(unittest.TestCase):

    def setUp(self):
        self.arr = copy.copy(MAT)
        self.matrix = Matrix(self.arr)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        matrix = Matrix(self.arr)
        self.assertTrue(np.all(matrix.arr == self.arr))

    def testMakeTrinaryMatrix(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            try:
                arr = Matrix.makeTrinaryMatrix(3, 3, prob0=0.9)
            except RuntimeError:
                continue
            arr_sq = arr*arr
            is_nozerorow = np.all(arr_sq.sum(axis=1) > 0)
            is_nozerocol = np.all(arr_sq.sum(axis=0) > 0)
            self.assertTrue(is_nozerorow and is_nozerocol)
            break

    def testRandomize(self):
        #if IGNORE_TEST:
        #    return
        matrix = Matrix(Matrix.makeTrinaryMatrix(4, 5, prob0=0.2))
        row_triples = []
        for idx in range(matrix.nrow):
            triple = (np.sum(matrix.arr[idx, :] < 0),
                      np.sum(matrix.arr[idx, :] == 0),
                      np.sum(matrix.arr[idx, :] > 0 ))
            row_triples.append(triple)


        

if __name__ == '__main__':
    unittest.main()