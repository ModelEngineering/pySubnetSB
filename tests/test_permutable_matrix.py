from sirn.permutable_matrix import PermutableMatrix # type: ignore
from sirn.matrix import Matrix # type: ignore

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


#############################
# Tests
#############################
class TestPermutableMatrixSerialization(unittest.TestCase):

    def setUp(self):
        raise NotImplementedError

    def testConstructor(self):
        raise NotImplementedError

    def testMakeDataFrame(self):
        raise NotImplementedError


class TestMatrixClassifier(unittest.TestCase):

    def setUp(self):
        pass

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.permutable_matrix = PermutableMatrix(MAT)
        self.assertTrue(np.all(self.permutable_matrix.array == MAT))
        self.assertTrue(isinstance(self.permutable_matrix, PermutableMatrix))

    def testNotIdenticalClassifications(self):
        # Tests if permuted matrices don't have the same classification
        # The test assumes that samples are very unlikely to be identical, but it can happen
        if IGNORE_TEST:
            return
        def test(num_iteration=100, size=5, prob0=1/3):
            for _ in range(num_iteration):
                arr = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                arr2 = Matrix.makeTrinaryMatrix(size, size, prob0=prob0)
                permutable_matrix = PermutableMatrix(arr)
                permutable_matrix2 = PermutableMatrix(arr2)
                self.assertNotEqual(permutable_matrix.hash_val, permutable_matrix2.hash_val)
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
                permutable_matrix = PermutableMatrix(arr)
                row_perm = np.random.permutation(range(size))
                col_perm = np.random.permutation(range(size))
                arr2 = arr.copy()
                arr2 = arr2[row_perm, :]
                arr2 = arr2[:, col_perm]
                permutable_matrix2 = PermutableMatrix(arr2)
                self.assertEqual(permutable_matrix.hash_val, permutable_matrix2.hash_val)
                self.assertTrue(permutable_matrix.isCompatible(permutable_matrix2))
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
                permutable_matrix = PermutableMatrix(PermutableMatrix.makeTrinaryMatrix(size, size))
                hash_val = permutable_matrix.hash_val
                if hash_val not in hash_dct:
                    hash_dct[hash_val] = []
                hash_dct[hash_val].append(permutable_matrix)
            lengths = np.array([len(v) for v in hash_dct.values()])
            frac_collision = np.sum([(l/num_iteration)**2 for l in lengths])
            self.assertLess(frac_collision, BOUNDS[size])
        #
        test(10000, 3)
        test(10000, 4)
        test(10000, 5)

    def testEq1(self):
        if IGNORE_TEST:
            return
        size = 3
        arr1 = PermutableMatrix.makeTrinaryMatrix(size, size, prob0=0.8)
        arr1 = MAT.copy()
        arr2 = arr1.copy()
        row_perm =  np.array([1, 0, 2])
        arr2 = arr2[row_perm, :]
        permutable_matrix1 = PermutableMatrix(arr1)
        permutable_matrix2 = PermutableMatrix(arr2)
        self.assertTrue(permutable_matrix1 == permutable_matrix2)

    def testEq2(self):
        if IGNORE_TEST:
            return
        arr1 = np.array([[ 1, -1, -1],
            [ 0,  0,  1],
            [ 1,  1,  0]])
        arr2 = np.array([[ 1, -1, -1],
            [ 1,  0,  1],
            [ 0,  1,  0]])
        permutable_matrix1 = PermutableMatrix(arr1)
        permutable_matrix2 = PermutableMatrix(arr2)
        if not permutable_matrix1 == permutable_matrix2:
            import pdb; pdb.set_trace()

    def testEq3(self):
        # Test permutably identical matrices
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=20):
            for _ in range(num_iteration):
                arr1 = PermutableMatrix.makeTrinaryMatrix(size, size)
                arr2 = arr1.copy()
                for _ in range(10):
                    perm =  np.random.permutation(range(size))
                    if np.sum(np.diff(perm)) != size - 1:  # Avoid the identity permutation
                        break
                arr2 = arr2[perm, :]
                arr2 = arr2[:, perm]
                permutable_matrix1 = PermutableMatrix(arr1)
                permutable_matrix2 = PermutableMatrix(arr2)
                self.assertTrue(permutable_matrix1 == permutable_matrix2)
        #
        test(3)
        test(10)
        test(20)
    
    def testEq4(self):
        # Test not permutably identical matrices
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=20):
            for _ in range(num_iteration):
                arr1 = PermutableMatrix.makeTrinaryMatrix(size, size)
                arr2 = arr1.copy()
                # Randomly change a value
                irow = np.random.randint(size)
                icol = np.random.randint(size)
                arr2[irow, icol] = -arr2[irow, icol]
                if arr2[irow, icol] == 0:
                    arr2[irow, icol] = 1
                # Construct the ordered matrices
                permutable_matrix1 = PermutableMatrix(arr1)
                permutable_matrix2 = PermutableMatrix(arr2)
                self.assertFalse(permutable_matrix1 == permutable_matrix2)
        #
        test(3)
        test(10)
        test(20)
        test(200)



if __name__ == '__main__':
    unittest.main()