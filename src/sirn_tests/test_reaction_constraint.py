from sirn.constraint import NULL_NMAT  # type: ignore
from sirn.reaction_constraint import ReactionConstraint # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore
from sirn.network import Network # type: ignore

import numpy as np
import re
import unittest


IGNORE_TEST = False
IS_PLOT = False
REACTANT_MATRIX = NamedMatrix(np.array([[1, 0], [0, 1], [0, 0]]))
PRODUCT_MATRIX = NamedMatrix(np.array([[1, 1], [1, 0], [0, 0]]))
#PRODUCT_MATRIX = NamedMatrix(np.array([[0, 1], [1, 0], [0, 0]]))


#############################
# Tests
#############################
class TestReactionConstraint(unittest.TestCase):

    def setUp(self):
        self.constraint = ReactionConstraint(REACTANT_MATRIX.copy(), PRODUCT_MATRIX.copy()) 

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.constraint.reactant_nmat, REACTANT_MATRIX)
        self.assertEqual(self.constraint.product_nmat, PRODUCT_MATRIX)
        self.assertEqual(self.constraint._categorical_nmat, NULL_NMAT)
        self.assertEqual(self.constraint._enumerated_nmat, NULL_NMAT)

    def testMakeReactionClassificationConstraintMatrix(self):
        if IGNORE_TEST:
            return
        size = 20
        network = Network.makeRandomNetworkByReactionType(size, size)
        constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat)
        named_matrix = constraint._makeClassificationConstraintMatrix()
        self.assertTrue(isinstance(named_matrix, NamedMatrix))
        df = named_matrix.dataframe
        self.assertEqual(len(df), size)
    
    def testMakePredecessorSuccessorConstraintMatrix(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            size = 30
            network = Network.makeRandomNetworkByReactionType(size, size)
            constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat)
            named_matrix = constraint._makeSuccessorConstraintMatrix()
            sum_arr = named_matrix.values.sum(axis=1)
            self.assertTrue(np.all(sum_arr <= size))

    def testMakeAutocatalysisConstraintMatrix(self):
        if IGNORE_TEST:
            return
        for _ in range(100):
            size = 10  # Doesn't work for two digit species
            network = Network.makeRandomNetworkByReactionType(size, size)
            constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat)
            named_matrix = constraint._makeAutocatalysisConstraintMatrix()
            self.assertTrue(isinstance(named_matrix, NamedMatrix))
            df = named_matrix.dataframe
            self.assertGreater(len(df), 0)
            # Verify the count of autocatalysis reactions
            num_autocatalysis = np.sum([len(re.findall("S" + str(n) + " .*->.*S" + str(n), str(network)))
                  for n in range(size)])
            # Regular expression counts occurrences, not reactions and so may exceed named_matrix sum.
            self.assertLessEqual(df.values.sum(), num_autocatalysis)

    def testcategoricalAndenumeratedConstraints(self):
        if IGNORE_TEST:
            return
        for _ in range(4):
            self.constraint.setSubset(True)
            self.assertTrue(self.constraint.equality_nmat is not NULL_NMAT)
            self.assertTrue(self.constraint.inequality_nmat is not NULL_NMAT)
            #
            self.constraint.setSubset(False)
            self.assertTrue(self.constraint.equality_nmat is not NULL_NMAT)
            self.assertTrue(self.constraint.inequality_nmat is NULL_NMAT)


if __name__ == '__main__':
    unittest.main()