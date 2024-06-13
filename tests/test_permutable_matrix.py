from sirn.permutable_matrix import PermutableMatrix, PermutableMatrixSerialization # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn import util # type: ignore

import pandas as pd  # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
MODEL_NAME = 'model_name'
SERIALIZATIONS = [PermutableMatrixSerialization(MODEL_NAME,
                                                np.random.randint(0, 2, (3, 3)),
                                                ['r1', 'r2', 'r3'],
                                                ['s1', 's2', 's3']) for _ in range(10)]


#############################
# Tests
#############################
class TestPermutableMatrixSerialization(unittest.TestCase):

    def setUp(self):
        self.serialization = PermutableMatrixSerialization(MODEL_NAME, MAT, ['r1', 'r2', 'r3'], ['s1', 's2', 's3']) 

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.all(self.serialization.array == MAT))

    def testRepr(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(str(self.serialization), str))
        stg = str(self.serialization)
        stg_arr = eval(stg)
        self.assertTrue(isinstance(eval(stg_arr[1]), list))

    def testMakeDataFrame(self):
        if IGNORE_TEST:
            return
        df = PermutableMatrixSerialization.makeDataFrame(SERIALIZATIONS)
        TYPES = {'model_name': str, 'array': np.ndarray, 'row_names': list, 'column_names': list}
        array = df.loc[0, "array"]
        self.assertTrue(isinstance(array, np.ndarray))
        #
        for column in df.columns:
            self.assertTrue(isinstance(df.loc[0, column], TYPES[column]))


class TestMatrixClassifier(unittest.TestCase):

    def setUp(self):
        pass

    def testConstructor(self):
        if IGNORE_TEST:
            return
        permutable_matrix = PermutableMatrix(MAT)
        self.assertTrue(np.all(permutable_matrix.array == MAT))
        self.assertTrue(isinstance(permutable_matrix, PermutableMatrix))

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

    def testIsPermutablyIdentical1(self):
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
        self.assertTrue(permutable_matrix1.isPermutablyIdentical(permutable_matrix2))

    def testIsPermutablyIdentical2(self):
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
        if not permutable_matrix1.isPermutablyIdentical(permutable_matrix2):
            import pdb; pdb.set_trace()

    def testIsPermutablyIdentical3(self):
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
                self.assertTrue(permutable_matrix1.isPermutablyIdentical(permutable_matrix2))
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
                self.assertFalse(permutable_matrix1.isPermutablyIdentical(permutable_matrix2))
        #
        test(3)
        test(10)
        test(20)
    
    def testSerializeOne(self):
        if IGNORE_TEST:
            return
        permutable_matrix = PermutableMatrix(MAT)
        serialization = permutable_matrix._serializeOne()
        self.assertTrue(isinstance(serialization, PermutableMatrixSerialization))
        self.assertTrue(np.all(serialization.array == permutable_matrix.array))
        self.assertTrue(isinstance(serialization.row_names, list))
        self.assertTrue(isinstance(serialization.column_names, list))
        self.assertTrue(isinstance(serialization.model_name, str))

    def testSerializeManyAndDeserialize(self):
        if IGNORE_TEST:
           return
        def test(size=3, num_mat=20, num_iteration=10):
            for _ in range(num_iteration):
                permutable_matrices = [PermutableMatrix(PermutableMatrix.makeTrinaryMatrix(size, size))
                                       for _ in range(num_mat)]
                df = PermutableMatrix.serializeMany(permutable_matrices)
                self.assertTrue(isinstance(df, pd.DataFrame))
                self.assertTrue(isinstance(df.loc[0, 'array'], np.ndarray))
                self.assertEqual(len(df), num_mat)
                #
                permutable_matrices2 = PermutableMatrix.deserializeDataFrame(df)
                trues = [pm1 == pm2 for pm1, pm2 in zip(permutable_matrices, permutable_matrices2)]
                self.assertTrue(all(trues))
        #
        test()
        test(size=20)

    def testEq(self):
        if IGNORE_TEST:
            return
        permutable_matrix = PermutableMatrix(MAT)
        self.assertTrue(permutable_matrix == permutable_matrix)
        # Test different matrices
        permutable_matrix2 = PermutableMatrix(MAT)
        permutable_matrix2.array[0, 0] = 2
        self.assertFalse(permutable_matrix == permutable_matrix2)

    def testDeserializeCSV(self):
        if IGNORE_TEST:
            return
        raise NotImplementedError
    
    def testDeserializeAntimonyFile(self):
        if IGNORE_TEST:
            return
        raise NotImplementedError
    
    def testDeserializeAntimonyDirectory(self):
        if IGNORE_TEST:
            return
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()