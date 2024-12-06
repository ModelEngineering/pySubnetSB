from pySubnetSB.named_matrix import NamedMatrix  # type: ignore

import copy
import numpy as np
import unittest
import time


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
MAT2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


#############################
# Tests
#############################
class TestNamedMatrix(unittest.TestCase):

    def setUp(self):
        self.array = copy.copy(MAT)
        self.named_matrix = NamedMatrix(MAT.copy(), row_names=[(1, 0), (0, 1), (0, 0)],
                                        column_names=['d', 'e', 'f'])

    def testConstructor(self):
        if IGNORE_TEST:
            return
        row_description = "rows"
        column_description = "columns"
        named_matrix = NamedMatrix(MAT2.copy(), row_names=[(1, 0), (0, 1), (0, 0), (0, 3)],
                                        column_names=['d', 'e', 'f'],
                                    row_description=row_description, column_description=column_description)
        self.assertTrue(np.all(named_matrix.values == MAT2))
        self.assertTrue("e" in str(named_matrix))
        self.assertTrue(row_description in str(named_matrix))
        self.assertTrue(column_description in str(named_matrix))

    def testPerformance0(self):
        if IGNORE_TEST:
            return
        arr = np.random.randint(-10, 10, (100, 100))
        start_time = time.time()
        for _ in range(10000):
            named_matrix = NamedMatrix(arr)
        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, 1e-1)

    def testEq(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.named_matrix == self.named_matrix)
        #
        named_matrix = NamedMatrix(MAT2.copy(), row_names=[(1, 0), (0, 1), (0, 0), (0, 3)],
                                        column_names=['d', 'e', 'f'])
        self.assertFalse(self.named_matrix == named_matrix)

    def testLe(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.named_matrix <= self.named_matrix)
        #
        named_matrix = NamedMatrix(MAT2.copy(), row_names=[(1, 0), (0, 1), (0, 0), (0, 3)],
                                        column_names=['d', 'e', 'f'])
        self.assertFalse(self.named_matrix <= named_matrix)
        #
        mat = MAT.copy()
        mat[0, 0] = 10
        named_matrix = NamedMatrix(mat, row_names=[(1, 0), (0, 1), (0, 0)],
                                        column_names=['d', 'e', 'f'])
        self.assertFalse(named_matrix <= self.named_matrix)

    def testTemplate(self):
        if IGNORE_TEST:
            return
        named_matrix = self.named_matrix.template()
        self.assertTrue(named_matrix == self.named_matrix)
        #
        mat = MAT.copy()
        mat[0, 0] = 10
        named_matrix = self.named_matrix.template(matrix=mat)
        self.assertFalse(named_matrix == self.named_matrix)
        self.assertTrue(named_matrix.isCompatible(self.named_matrix))

    def testGetSubNamedMatrix(self):
        if IGNORE_TEST:
            return
        result = self.named_matrix.getSubNamedMatrix(row_names=[(1, 0), (0, 1)], column_names=['d', 'e'])
        named_matrix = result.named_matrix
        self.assertTrue(np.all(named_matrix.values == np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.all(named_matrix.row_names == np.array([(1, 0), (0, 1)])))
        self.assertTrue(np.all(named_matrix.column_names == np.array(['d', 'e'])))

    def testGetSubMatrix(self):
        if IGNORE_TEST:
            return
        matrix = self.named_matrix.getSubMatrix(row_idxs=range(2), column_idxs=range(2))
        self.assertTrue(np.all(matrix.values == np.array([[1, 0], [0, 1]])))

    def testPerformance(self):
        if IGNORE_TEST:
            return
        num_row = 100
        num_column = 100
        mat = np.random.randint(-10, 10, (num_row, num_column))
        named_matrix = NamedMatrix(mat)
        def timeit(num_iteration=1000, is_named_matrix=True):
            t0 = time.time()
            for _ in range(num_iteration):
                if is_named_matrix:
                    _ = named_matrix.getSubNamedMatrix(row_names=range(10), column_names=range(10))
                else:
                    _ = named_matrix.getSubMatrix(row_idxs=range(10), column_idxs=range(10))
            t1 = time.time()
            return t1 - t0
        #
        time_named_matrix = timeit(is_named_matrix=True)
        time_matrix = timeit(is_named_matrix=False)
        self.assertTrue(time_named_matrix/time_matrix > 100)
        #print(f"TimeNamed: {time_named_matrix}", f"TimeMatrix: {time_matrix}")

    def testCopyEquals(self):
        if IGNORE_TEST:
            return
        named_matrix = self.named_matrix.copy()
        self.assertTrue(named_matrix == self.named_matrix)

    def testTranspose(self):
        if IGNORE_TEST:
            return
        named_matrix = self.named_matrix.transpose()
        reverted = named_matrix.transpose()
        self.assertTrue(reverted == self.named_matrix)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        string = self.named_matrix.serialize()
        named_matrix = NamedMatrix.deserialize(string)
        self.assertTrue(named_matrix == self.named_matrix)

    def testHstack(self):
        if IGNORE_TEST:
            return
        named_matrix = NamedMatrix(np.array([[1, 0], [0, 1], [0, 0]]), column_names=['a', 'b'])
        named_matrix2 = NamedMatrix(np.array([[1], [0], [0]]), column_names=['c'])
        # Same shapes
        result = NamedMatrix.hstack([named_matrix, named_matrix2])
        expected_arr = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]])
        self.assertTrue(np.all(result.values == expected_arr))
        self.assertTrue(np.all(result.column_names == np.array(['a', 'b', 'c'])))
        self.assertEqual(result.num_column, named_matrix.num_column + named_matrix2.num_column)
        # Different shapes
        with self.assertRaises(ValueError):
            NamedMatrix.hstack([named_matrix, named_matrix2.transpose()])
        # Different row names
        with self.assertRaises(ValueError):
            named_matrix2 = NamedMatrix(np.array([[1], [0], [0]]), column_names=['c'],
                                        row_names=['x', 'y', 'z'])
            _ = NamedMatrix.hstack([named_matrix, named_matrix2])

    def testVstack(self):
        if IGNORE_TEST:
            return
        array1 =np.array([[1, 0], [0, 1], [0, 0]])
        array2 =np.array([[1, 2], [0, 2]])
        named_matrix1 = NamedMatrix(array1, row_names=['a', 'b', 'c'])
        named_matrix2 = NamedMatrix(array2, row_names=['d', 'e'])
        # Same shapes
        result = NamedMatrix.vstack([named_matrix1, named_matrix2])
        self.assertTrue(np.all(result.values == np.vstack([array1, array2])))
        self.assertTrue(np.all(result.row_names == np.array(['a', 'b', 'c', 'd', 'e'])))
        self.assertEqual(result.num_row, named_matrix1.num_row + named_matrix2.num_row)
        # Different shapes
        with self.assertRaises(ValueError):
            NamedMatrix.vstack([named_matrix1, named_matrix2.transpose()])
        # Different row names
        with self.assertRaises(ValueError):
            named_matrix2 = NamedMatrix(np.array([[1], [0], [0]]),
                                        column_names=['x', 'y'])
            _ = NamedMatrix.hstack([named_matrix1, named_matrix2])
        

if __name__ == '__main__':
    unittest.main(failfast=True)