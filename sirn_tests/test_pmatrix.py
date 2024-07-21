from sirn.pmatrix import PMatrix # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn import util # type: ignore
import sirn.constants as cn  # type: ignore
import sirn.util as util  # type: ignore

import pandas as pd  # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
util.IS_TIMEIT = False  # Set to True for timing tests
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class TestPMatrix(unittest.TestCase):

    def setUp(self):
        array = MAT.copy()
        self.pmatrix = PMatrix(array)

    @util.timeit
    def testConstructor(self):
        if IGNORE_TEST:
            return
        pmatrix = PMatrix(MAT)
        self.assertTrue(np.all(pmatrix.array == MAT))
        self.assertTrue(isinstance(pmatrix, PMatrix))

    @util.timeit
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

    @util.timeit
    def testIdenticalClassifications(self):
        # Tests if permuted matrices have the same encodings
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
    
    @util.timeit
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

    @util.timeit
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
        self.assertTrue(pmatrix1.isPermutablyIdentical(pmatrix2, is_sirn=False))

    @util.timeit
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
        self.assertTrue(pmatrix1.isPermutablyIdentical(pmatrix2, is_sirn=False))

    @util.timeit
    def testIsPermutablyIdentical3(self):
        # Test permutably identical matrices and calculation limits
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=2, expected_result=True):
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
                result = pmatrix1.isPermutablyIdentical(pmatrix2, 
                        is_find_all_perms=True, max_num_perm=10000)
                if expected_result:
                    self.assertTrue(result)
                else:
                    self.assertFalse(result)
        #
        test(size=3)
        test(size=200, expected_result=False)
        test(size=10)
    
    @util.timeit
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

    @util.timeit
    def testIsPermutablyIdenticalNotSIRN(self):
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
        result = pmatrix1.isPermutablyIdentical(pmatrix2, is_sirn=False)
        self.assertTrue(result)

    @util.timeit
    def testIsPermutablyIdenticalNotSIRN2(self):
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
                self.assertFalse(pmatrix1.isPermutablyIdentical(pmatrix2, is_sirn=False))
        #
        test(3)
        test(5)

    @util.timeit
    def testIsPermutablyIdenticalNotSIRN3(self):
        # Test permutably identical matrices and calculation limits
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=10, expected_result=True):
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
                result = pmatrix1.isPermutablyIdentical(pmatrix2, is_sirn=False,
                        is_find_all_perms=True, max_num_perm=1000)
                if expected_result:
                    self.assertTrue(result or result.is_excessive_perm)
                else:
                    self.assertFalse(result)
        #
        test(size=200, expected_result=False)
        test(3)
        test(10)

    @util.timeit
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

    @util.timeit
    def testRandomize(self):
        if IGNORE_TEST:
            return
        is_done = False
        for _ in range(10):
            pmatrix = self.pmatrix.randomize().pmatrix
            if pmatrix != self.pmatrix:
                is_done = True
                break
        if not is_done:
            raise ValueError("Test failed")
        self.assertTrue(self.pmatrix.isPermutablyIdentical(pmatrix))

    @util.timeit
    def testLogEstimate(self):
        if IGNORE_TEST:
            return
        pmatrix = PMatrix(MAT)
        self.assertTrue(np.isclose(pmatrix.log_estimate, 2*np.log10(6)))

    @util.timeit
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
        max_log_perm = test(15, 2000)
        self.assertGreater(max_log_perm, 4)

    @util.timeit
    def testMaxPerm(self):
        if IGNORE_TEST:
            return
        def test(max_num_perm):
            is_done = False
            # Should find a non-identical matrix
            for _ in range(10):
                pmatrix = self.pmatrix.randomize().pmatrix
                result = self.pmatrix.isPermutablyIdentical(
                    pmatrix,
                    max_num_perm=0)
                if not result.is_permutably_identical:
                    is_done = True
                    break
            if is_done:
                self.assertTrue(True)
            else:
                self.assertTrue(False)
        #
        test(0)
        test(1)
        test(4)

    @util.timeit
    def testIsPermutablyIdenticalSubset1(self):
        # Test permutably identical matrices and calculation limits
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=1, subset_size=2, max_num_perm=1000, expected_result=True):
            if subset_size > size:
                raise ValueError("subset_size must be less than size")
            for _ in range(num_iteration):
                arr1 = PMatrix.makeTrinaryMatrix(size, size)
                arr2 = arr1.copy()
                for _ in range(10):
                    perm =  np.random.permutation(range(size))
                    if np.sum(np.diff(perm)) != size - 1:  # Avoid the identity permutation
                        break
                arr2 = arr2[perm, :]
                arr2 = arr2[:, perm]
                arr2 = arr2[:subset_size, :subset_size]
                pmatrix1 = PMatrix(arr1)
                pmatrix2 = PMatrix(arr2)
                result = pmatrix2.isPermutablyIdenticalSubset(pmatrix1, 
                        max_num_perm=max_num_perm)
                self.assertTrue(bool(result) == expected_result)
        #
        test(size=8, subset_size=3, num_iteration=1, max_num_perm=10000)
        test(size=3, subset_size=2, num_iteration=5)

    @util.timeit
    def testIsPermutablyIdenticalSubsetNot(self):
        # Test not permutably identical subset
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=10, subset_size=2):
            if subset_size > size:
                raise ValueError("subset_size must be less than size")
            for _ in range(num_iteration):
                arr1 = PMatrix.makeTrinaryMatrix(size, size)
                arr2 = arr1.copy()
                for _ in range(10):
                    perm =  np.random.permutation(range(size))
                    if np.sum(np.diff(perm)) != size - 1:  # Avoid the identity permutation
                        break
                arr2 = arr2[perm, :]
                arr2 = arr2[:, perm]
                arr2 = arr2[:subset_size, :subset_size]
                # Randomly change a value
                irow = np.random.randint(subset_size)
                icol = np.random.randint(subset_size)
                arr2[irow, icol] = 2
                #  Construct the ordered matrices
                pmatrix1 = PMatrix(arr1)
                pmatrix2 = PMatrix(arr2)
                result = pmatrix2.isPermutablyIdenticalSubset(pmatrix1, 
                        max_num_perm=10000)
                self.assertFalse(result)
        #
        test(3)
        test(250, 5)


if __name__ == '__main__':
    unittest.main(failfast=True)