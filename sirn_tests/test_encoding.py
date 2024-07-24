from sirn.encoding import Encoding, NamedMatrix, CRITERIA_PAIRED_VALUES, CRITERIA_PAIRED_NAMES  # type: ignore
from sirn.matrix import Matrix # type: ignore

import pandas as pd # type: ignore
import numpy as np
import time
import unittest


IGNORE_TEST = True
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])


#############################
# Tests
#############################
class TestEncoding(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.encoding = Encoding(MAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.encoding = Encoding(MAT)
        self.assertTrue(np.all(self.encoding.collection == MAT))
        self.assertTrue(isinstance(self.encoding, Encoding))
        self.assertTrue(np.all(self.encoding.encoding_mat[:, 3] == np.array([2, 2, 1])))
        self.assertEqual(len(self.encoding.encoding_dct), 2)

    def testTimingMakeEncodingMat(self):
        if IGNORE_TEST:
            return
        start = time.time()
        count = 0
        for _ in range(100):
            count += 1
            mat = np.random.randint(0, 2, (10, 10))
            _ = Encoding(mat)
        end = time.time()
        self.assertLess(end - start, 5)

    def testEq(self):
        if IGNORE_TEST:
            return
        encoding = Encoding(MAT)
        self.assertTrue(encoding == self.encoding)
        mat = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 2]])
        encoding = Encoding(mat)
        self.assertFalse(encoding == self.encoding)
    
    def testTimingEq(self):
        if IGNORE_TEST:
            return
        mat = np.random.randint(0, 2, (100, 100))
        encoding1 = Encoding(mat)
        encoding2 = Encoding(mat)
        start = time.time()
        for _ in range(10000):
            _ = encoding1 == encoding2
        end = time.time()
        self.assertLess(end - start, 1)

    def testEncodeCollection(self):
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        encoded_array = encoding._encodeCollection()
        self.assertTrue(np.all(np.shape(encoded_array) == np.shape(encoding.collection)))
        diff = set(encoded_array.flatten()).symmetric_difference(range(1, 8))
        self.assertEqual(len(diff), 0)

    def testMakeEncodingMat(self):
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        encoding_mat, encodings = encoding._makeEncodingMat()
        self.assertTrue(np.all(np.shape(encoding_mat) == (4, 7)))
        self.assertEqual(len(encodings), np.shape(encoding_mat)[0])

    def testMakeAllPairEncodingDataFrame(self):
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        encoding_df = encoding._makeAllPairEncodingDataFrame()
        self.assertTrue(isinstance(encoding_df, pd.DataFrame))
        trues = [(len(v) == 3) and ("-" in v) for v in encoding_df.columns]
        self.assertTrue(all(trues))
        trues = [len(v) == 2 for v in encoding_df.index]
        self.assertTrue(all(trues))

    def testMakeAllPairEncodingMatrix(self):
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        named_matrix = encoding._makeAllPairEncodingMatrix()
        matrix = named_matrix.matrix
        self.assertTrue(isinstance(matrix, np.ndarray))

    def testMakeAdjacentPairEncodingDataFrame(self):
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        encoding_df = encoding._makeAdjacentPairEncodingDataFrame()
        self.assertTrue(isinstance(encoding_df, pd.DataFrame))
        trues = [(len(v) == 3) and ("-" in v) for v in encoding_df.columns]
        self.assertTrue(all(trues))
        trues = [len(v) == 2 for v in encoding_df.index]
        self.assertTrue(all(trues))
    
    def testMakeAdjacentPairEncodingMatrix(self):
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        encoding_mat = encoding._makeAdjacentPairEncodingMatrix()
        index_mat = encoding_mat.matrix > 0
        self.assertTrue(isinstance(encoding_mat, NamedMatrix))
        self.assertEqual(np.sum(index_mat[0, :]), 3)

    def testMakeAdjacentPairEncodingMatrixScale(self):
        if IGNORE_TEST:
            return
        def test(size=3, num_iteration=10):
            for _ in range(num_iteration):
                mat = Matrix.makeTrinaryMatrix(size, size)
                encoding = Encoding(mat)
                encoding_mat = encoding._makeAdjacentPairEncodingMatrix()
                self.assertTrue(isinstance(encoding_mat, NamedMatrix))
        #
        test(size=5)
        test(size=70, num_iteration=2)
    
    def testPairEncodingMatrices(self):
        # Test if the adjacent pair is consistent with All pairs
        #if IGNORE_TEST:
        #    return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        adj_named = encoding._makeAdjacentPairEncodingMatrix()
        all_named = encoding._makeAllPairEncodingMatrix()
        index_arr = np.array([i for i, v in enumerate(all_named.row_names) if v in adj_named.row_names])
        import pdb; pdb.set_trace()

    def testPairEncodings(self):
        # Test if the adjacent pair is consistent with All pairs
        if IGNORE_TEST:
            return
        mat = np.array([[1, 0, 0], [-.1, .1, 0], [-2, 1, 1], [-1, 1, 2]])
        encoding = Encoding(mat)
        adj_df = encoding._makeAdjacentPairEncodingDataFrame()
        all_df = encoding._makeAllPairEncodingDataFrame()
        reduced_all_df = all_df.loc[adj_df.index, adj_df.columns].copy()
        self.assertTrue(np.all(adj_df == reduced_all_df))

    def testPairEncodings2(self):
        # Test if the adjacent pair is consistent with All pairs
        if IGNORE_TEST:
            return
        def test(size:int, num_iter:int=10):
            for _ in range(num_iter):
                mat = Matrix.makeTrinaryMatrix(size, size)
                encoding = Encoding(mat)
                adj_df = encoding.adjacent_pair_encoding_df
                all_df = encoding.all_pair_encoding_df
                reduced_all_df = all_df.loc[adj_df.index, adj_df.columns].copy()
                self.assertTrue(np.all(adj_df == reduced_all_df))
        #
        test(5)
        test(20, num_iter=2)
            
        


if __name__ == '__main__':
    unittest.main()