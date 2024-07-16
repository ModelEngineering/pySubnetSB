from sirn.array_collection import ArrayCollection # type: ignore
from sirn.matrix import Matrix # type: ignore

import numpy as np
import scipy  # type: ignore
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
        self.assertGreater(self.collection.encodings[-1], self.collection.encodings[0])

    def testClassifyArray(self):
        if IGNORE_TEST:
            return
        sorted_mat = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]])
        self.assertTrue(np.allclose(self.collection.sorted_mat, sorted_mat))

    def testEncode2(self):
        # Test random sequences
        if IGNORE_TEST:
            return
        def test(num_iteration=10, size=10, prob0=1/3):
            # Makes a trinary matrix and checks the encodings
            for _ in range(num_iteration):
                matrix = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                arr = matrix[1,:]
                counts = []
                counts.append(np.sum(arr < 0))
                counts.append(np.sum(arr == 0))
                counts.append(np.sum(arr > 0))
                collection = ArrayCollection(matrix)
                new_encoding = 0
                for idx in range(3):
                    new_encoding += counts[idx]*1000**idx
                self.assertGreaterEqual(size, collection.num_partition)
        #
        test()
        test(prob0=2/3)
        test(prob0=1/5)

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
                #print(collection.num_partition/size)
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

    def testSubsetIterator(self):
        if IGNORE_TEST:
            return
        if False:
            mat = np.array([[1, 0], [1, 1]])
            collection = ArrayCollection(mat)
            subsets = list(collection.subsetIterator(self.collection))
            self.assertTrue(len(subsets) == 1)
            self.assertTrue(len(subsets[0]) == len(mat))
        #
        mat = np.array([[1, 0], [0, 1]])
        collection = ArrayCollection(mat)
        subsets = list(collection.subsetIterator(self.collection))
        self.assertTrue(len(subsets) == 2)
        import pdb; pdb.set_trace()
        self.assertTrue(len(subsets[0]) == len(mat))


if __name__ == '__main__':
    unittest.main()