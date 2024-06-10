from sirn.ordered_matrix import OrderedMatrix # type: ignore
from sirn.matrix import Matrix # type: ignore

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
        self.ordered_matrix = OrderedMatrix(MAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.all(self.ordered_matrix.arr == MAT))
        self.assertTrue(isinstance(self.ordered_matrix, OrderedMatrix))

    def testNotIdenticalClassifications(self):
        # Tests if permuted matrices don't have the same classification
        # The test assumes that samples are very unlikely to be identical, but it can happen
        if IGNORE_TEST:
            return
        def test(num_iteration=100, size=5, prob0=1/3):
            for _ in range(num_iteration):
                arr = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                arr2 = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                ordered_matrix = OrderedMatrix(arr)
                ordered_matrix2 = OrderedMatrix(arr2)
                self.assertNotEqual(ordered_matrix.hash_val, ordered_matrix2.hash_val)
        #
        test()
        test(size=10, prob0=0.2)

    def testIdenticalClassifications(self):
        # Tests if permuted matrices have the same classification
        if IGNORE_TEST:
            return
        def test(num_iteration=100, size=5, prob0=1/3):
            for _ in range(num_iteration):
                arr = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                ordered_matrix = OrderedMatrix(arr)
                row_perm = np.random.permutation(range(size))
                col_perm = np.random.permutation(range(size))
                arr2 = arr.copy()
                arr2 = arr2[row_perm, :]
                arr2 = arr2[:, col_perm]
                ordered_matrix2 = OrderedMatrix(arr2)
                self.assertEqual(ordered_matrix.hash_val, ordered_matrix2.hash_val)
                self.assertTrue(ordered_matrix.isCompatible(ordered_matrix2))
        #
        test()
        test(size=10, prob0=0.2)
    
    def testHashCollision(self):
        # Evaluate the probability that two randomly chosen Ordered matrices have the same hash value
        if IGNORE_TEST:
            return
        BOUNDS = {3: 2e-2, 4: 1e-2, 5: 1e-3}
        def test(num_iteration, size):
            hash_dct = {}
            for _ in range(num_iteration):
                ordered_matrix = OrderedMatrix(OrderedMatrix.makeTrinaryMatrix(size, size))
                hash_val = ordered_matrix.hash_val
                if hash_val not in hash_dct:
                    hash_dct[hash_val] = []
                hash_dct[hash_val].append(ordered_matrix)
            lengths = np.array([len(v) for v in hash_dct.values()])
            frac_collision = np.sum([(l/num_iteration)**2 for l in lengths])
            self.assertLess(frac_collision, BOUNDS[size])
        #
        test(10000, 3)
        test(10000, 4)
        test(10000, 5)


        

if __name__ == '__main__':
    unittest.main()