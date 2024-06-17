from sirn.pmatrix import PMatrix # type: ignore
from sirn.pmatrix_collection import PMatrixCollection # type: ignore
from sirn.pmc_serializer import PMCSerializer # type: ignore
import sirn.constants as cn  # type: ignore

import os
import pandas as pd  # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
MODEL_NAME = 'model_name'
PMATRIX_COLLECTION = PMatrixCollection.makeRandomCollection(num_pmatrix=10, matrix_size=5)
SERIALIZER_PATH = os.path.join(cn.TEST_DIR, 'pmc_serializers.csv')
REMOVE_FILES = [SERIALIZER_PATH]
FILE_NAME = 'Alharbi2019_TNVM.ant'


#############################
# Tests
#############################
class TestPermutableSerializer(unittest.TestCase):

    def setUp(self):
        self.serializer = PMCSerializer(PMATRIX_COLLECTION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.serializer.collection.pmatrices), len(PMATRIX_COLLECTION))

    def testRepr(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(str(self.serializer), str))

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        def test(matrix_size=3, num_pmatrix=20, num_iteration=10):
            for _ in range(num_iteration):
                pmatrix_collection = PMatrixCollection.makeRandomCollection(num_pmatrix=num_pmatrix,
                                    matrix_size=matrix_size)
                serializer = PMCSerializer(pmatrix_collection)
                df = serializer.serialize()
                for column in df.columns:
                    if "num" in column:
                        if not isinstance(df.loc[0, column], np.int64):
                            import pdb; pdb.set_trace()
                        #self.assertTrue(isinstance(df.loc[0, column], np.int64))
                    else:
                        #self.assertTrue(isinstance(df.loc[0, column], str))
                        if not isinstance(df.loc[0, column], str):
                            import pdb; pdb.set_trace()
                # Deserialize
                pmatrix_collection = PMCSerializer.deserialize(df)
                self.assertTrue(isinstance(pmatrix_collection, PMatrixCollection))
                trues = [pm1 == pm2 for pm1, pm2 in 
                        zip(serializer.collection.pmatrices, pmatrix_collection.pmatrices)]
                self.assertTrue(all(trues))
        #
        test()
        test(matrix_size=1000, num_pmatrix=2, num_iteration=1)
        test(matrix_size=10, num_pmatrix=20, num_iteration=100)
    
    def testDeserializeAntimonyFile(self):
        if IGNORE_TEST:
            return
        model_path = os.path.join(cn.MODEL_DIR, FILE_NAME)
        pmatrix = PMCSerializer._makePMatrixAntimonyFile(model_path)
        self.assertEqual(pmatrix.num_row, 3) # 3  species
        self.assertEqual(pmatrix.num_column, 6) # 6 reactions
        self.assertTrue(isinstance(pmatrix, PMatrix))
        self.assertTrue(pmatrix.isPermutablyIdentical(pmatrix))
    
    def testDeserializeAntimonyDirectory(self):
        if IGNORE_TEST:
            return
        pmatrix_collection = PMCSerializer.makePMCollectionAntimonyDirectory(cn.MODEL_DIR)
        self.assertTrue(isinstance(pmatrix_collection, PMatrixCollection))
        self.assertTrue(len(pmatrix_collection) == 1)
        name = FILE_NAME.split(".")[0]
        self.assertTrue(name in str(pmatrix_collection))


if __name__ == '__main__':
    unittest.main()