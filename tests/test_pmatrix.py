from sirn.pmatrix import PMatrix # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn import util # type: ignore
import sirn.constants as cn  # type: ignore

import pandas as pd  # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class TestPMatrix(unittest.TestCase):

    def setUp(self):
        array = MAT.copy()
        self.pmatrix = PMatrix(array)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        pmatrix = PMatrix(MAT)
        self.assertTrue(np.all(pmatrix.array == MAT))
        self.assertTrue(isinstance(pmatrix, PMatrix))

    def testNotIdenticalClassifications(self):
        # Tests if permuted matrices don't have the same classification
        # The test assumes that samples are very unlikely to be identical, but it can happen
        if IGNORE_TEST:
            return
        def test(num_iteration=100, size=5, prob0=1/3):
            for _ in range(num_iteration):
                arr = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                arr2 = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                pmatrix = PMatrix(arr)
                pmatrix2 = PMatrix(arr2)
                self.assertNotEqual(pmatrix.hash_val, pmatrix2.hash_val)
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
                pmatrix = PMatrix(arr)
                row_perm = np.random.permutation(range(size))
                col_perm = np.random.permutation(range(size))
                arr2 = arr.copy()
                arr2 = arr2[row_perm, :]
                arr2 = arr2[:, col_perm]
                pmatrix2 = PMatrix(arr2)
                self.assertEqual(pmatrix.hash_val, pmatrix2.hash_val)
                self.assertTrue(pmatrix.isCompatible(pmatrix2))
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
                pmatrix = PMatrix(PMatrix.makeTrinaryMatrix(size, size))
                hash_val = pmatrix.hash_val
                if hash_val not in hash_dct:
                    hash_dct[hash_val] = []
                hash_dct[hash_val].append(pmatrix)
            lengths = np.array([len(v) for v in hash_dct.values()])
            frac_collision = np.sum([(l/num_iteration)**2 for l in lengths])
            self.assertLess(frac_collision, BOUNDS[size])
        #
        test(10000, 3)
        test(10000, 4)
        test(10000, 5)

    def testIsPermutablyIdentical1(self):
        if IGNORE_TEST:
            return
        size = 3
        arr1 = PMatrix.makeTrinaryMatrix(size, size, prob0=0.8)
        arr1 = MAT.copy()
        arr2 = arr1.copy()
        row_perm =  np.array([1, 0, 2])
        arr2 = arr2[row_perm, :]
        pmatrix1 = PMatrix(arr1)
        pmatrix2 = PMatrix(arr2)
        self.assertTrue(pmatrix1.isPermutablyIdentical(pmatrix2))

    def testIsPermutablyIdentical2(self):
        if IGNORE_TEST:
            return
        arr1 = np.array([[ 1, -1, -1],
            [ 0,  0,  1],
            [ 1,  1,  0]])
        arr2 = np.array([[ 1, -1, -1],
            [ 1,  0,  1],
            [ 0,  1,  0]])
        pmatrix1 = PMatrix(arr1)
        pmatrix2 = PMatrix(arr2)
        self.assertTrue(pmatrix1.isPermutablyIdentical(pmatrix2))

    def testIsPermutablyIdentical3(self):
        # Test permutably identical matrices
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=20):
            for _ in range(num_iteration):
                arr1 = PMatrix.makeTrinaryMatrix(size, size)
                arr2 = arr1.copy()
                for _ in range(10):
                    perm =  np.random.permutation(range(size))
                    if np.sum(np.diff(perm)) != size - 1:  # Avoid the identity permutation
                        break
                arr2 = arr2[perm, :]
                arr2 = arr2[:, perm]
                pmatrix1 = PMatrix(arr1)
                pmatrix2 = PMatrix(arr2)
                self.assertTrue(pmatrix1.isPermutablyIdentical(pmatrix2))
        #
        test(3)
        test(10)
        test(20)
    
    def testIsPermutablyIdentical4(self):
        # Test not permutably identical matrices
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=20):
            for _ in range(num_iteration):
                arr1 = PMatrix.makeTrinaryMatrix(size, size)
                arr2 = arr1.copy()
                # Randomly change a value
                irow = np.random.randint(size)
                icol = np.random.randint(size)
                arr2[irow, icol] = -arr2[irow, icol]
                if arr2[irow, icol] == 0:
                    arr2[irow, icol] = 1
                # Construct the ordered matrices
                pmatrix1 = PMatrix(arr1)
                pmatrix2 = PMatrix(arr2)
                self.assertFalse(pmatrix1.isPermutablyIdentical(pmatrix2))
        #
        test(3)
        test(10)
        test(20)

    def testEq(self):
        if IGNORE_TEST:
            return
        pmatrix = PMatrix(MAT)
        self.assertTrue(pmatrix == pmatrix)
        # Test different matrices
        pmatrix2 = PMatrix(MAT.copy())
        pmatrix2.array[0, 0] = 2
        result = pmatrix == pmatrix2
        self.assertFalse(result)

    def testRandomize(self):
        if IGNORE_TEST:
            return
        pmatrix = self.pmatrix.randomize().pmatrix
        import pdb;
        self.assertTrue(pmatrix != self.pmatrix)
        self.assertTrue(self.pmatrix.isPermutablyIdentical(pmatrix))

    def testLogEstimate(self):
        if IGNORE_TEST:
            return
        pmatrix = PMatrix(MAT)
        self.assertTrue(np.isclose(pmatrix.log_estimate, 2*np.log10(6)))

    def testLogEstimate2(self):
        if IGNORE_TEST:
            return
        log_estimates = []
        def test(size=3, num_iteration=20):
            for _ in range(num_iteration):
                arr = PMatrix.makeTrinaryMatrix(size, size)
                pmatrix = PMatrix(arr)
                log_estimates.append(pmatrix.log_estimate)
            return np.max(log_estimates)
        #
        max_log_num_permutation = test(15, 2000)
        self.assertGreater(max_log_num_permutation, 2)


if __name__ == '__main__':
    unittest.main()