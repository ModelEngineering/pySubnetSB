from sirn.array_collection import ArrayCollection # type: ignore
from sirn.matrix import Matrix # type: ignore

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])


#############################
# Tests
#############################
class TestArrayCollection(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.collection = ArrayCollection(MAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.all(self.collection.collection == MAT))
        self.assertTrue(isinstance(self.collection, ArrayCollection))
        self.assertTrue(np.all(self.collection.encoding.encoding_mat[0, :]  \
                               == self.collection.encoding.encoding_mat[1, :]))

    def testConstrainedPermutationIteratorSimple(self):
        if IGNORE_TEST:
            return
        self.collection = ArrayCollection(MAT)
        permutations = list(self.collection.constrainedPermutationIterator())
        for permutation in [ [0, 1, 2], [1, 0, 2]]:
            self.assertTrue(any([ np.allclose(permutation, p) for p in permutations]))
    
    def testConstrainedPermutationIteratorComplicated(self):
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=10, max_num_perm=int(1e3)):
            for _ in range(num_iteration):
                mat = Matrix.makeTrinaryMatrix(size, size)
                collection = ArrayCollection(mat)
                permutations = list(collection.constrainedPermutationIterator(max_num_perm=max_num_perm))
                # Check that each permutation is unique
                for idx, permutation in enumerate(permutations):
                    checks = list(permutations)
                    checks = checks[:idx] + checks[idx+1:]
                    self.assertFalse(any([np.allclose(permutation, p) for p in checks]))
        #
        test(size=5)
        test(size=15)
        test(size=20)

    def testSubsetIteratorTrue(self):
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0], [0, 1]])
        collection = ArrayCollection(mat)
        subsets = list(collection.subsetIterator(self.collection))
        self.assertTrue(len(subsets[0]) == len(mat))
        self.assertTrue(len(subsets) == len(MAT)*(len(MAT)-1))

    def testSubsetIteratorFalse(self):
        if IGNORE_TEST:
            return
        self.collection = ArrayCollection(MAT)
        mat = np.array([[2, 0], [0, 1]])
        collection = ArrayCollection(mat)
        subsets = list(collection.subsetIterator(self.collection))
        self.assertTrue(len(subsets) == 0)

    def testSubsetIteratorComplicated(self):
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=5):
            for _ in range(num_iteration):
                mat = Matrix.makeTrinaryMatrix(size, size)
                submat = mat[1:, 1:]
                collection = ArrayCollection(mat)
                subcollection = ArrayCollection(submat)
                subsets = list(subcollection.subsetIterator(collection))
                expected_arr = np.array(range(1, size))
                self.assertTrue(np.any([np.allclose(s, expected_arr) for s in subsets]))
        #
        test(size=5)
        test(size=15)
        test(size=20)


if __name__ == '__main__':
    unittest.main()