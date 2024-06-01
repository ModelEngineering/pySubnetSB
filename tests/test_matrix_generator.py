from sirn.matrix_generator import MatrixGenerator # type: ignore

import copy
import numpy as np
import unittest


IGNORE_TEST = True
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


#############################
# Tests
#############################
class TestMatrixClassifier(unittest.TestCase):

    def setUp(self):
        self.arr = copy.copy(MAT)
        self.matrix_generator = MatrixGenerator(self.arr)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        matrix_generator = MatrixGenerator(self.arr)
        self.assertTrue(np.all(matrix_generator.arr == self.arr))

    def testMakeTrinaryMatrix(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            try:
                matrix_generator = MatrixGenerator.makeTrinaryMatrix(3, 3, prob0=0.9)
            except RuntimeError:
                continue
            matrix_sq = matrix_generator.arr*matrix_generator.arr
            is_nozerorow = np.all(matrix_sq.sum(axis=1) > 0)
            is_nozerocol = np.all(matrix_sq.sum(axis=0) > 0)
            self.assertTrue(is_nozerorow and is_nozerocol)
            break

    def testRandomize(self):
        #if IGNORE_TEST:
        #    return
        matrix_generator = MatrixGenerator.makeTrinaryMatrix(4, 5, prob0=0.2)
        row_triples = []
        for idx in range(matrix_generator.nrow):
            triple = (np.sum(matrix_generator.arr[idx, :] < 0),
                      np.sum(matrix_generator.arr[idx, :] == 0),
                      np.sum(matrix_generator.arr[idx, :] > 0 ))
            row_triples.append(triple)
        import pdb; pdb.set_trace()


        

if __name__ == '__main__':
    unittest.main()