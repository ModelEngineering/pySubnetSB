from sirn.constraint import NULL_NMAT  # type: ignore
from sirn.species_constraint import SpeciesConstraint # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore
from sirn.network import Network # type: ignore

import numpy as np
import re
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
REACTANT_MATRIX = NamedMatrix(np.array([[1, 0], [0, 1], [0, 0]]))
PRODUCT_MATRIX = NamedMatrix(np.array([[1, 1], [1, 0], [0, 0]]))
#PRODUCT_MATRIX = NamedMatrix(np.array([[0, 1], [1, 0], [0, 0]]))


#############################
# Tests
#############################
class TestSpeciesConstraint(unittest.TestCase):

    def setUp(self):
        self.species_constraint = SpeciesConstraint(REACTANT_MATRIX.copy(), PRODUCT_MATRIX.copy()) 

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.species_constraint.reactant_nmat, REACTANT_MATRIX)
        self.assertEqual(self.species_constraint.product_nmat, PRODUCT_MATRIX)
        self.assertEqual(self.species_constraint._equality_nmat, NULL_NMAT)
        self.assertEqual(self.species_constraint._inequality_nmat, NULL_NMAT)

    def testMakeSpeciesConstraintMatrix(self):
        if IGNORE_TEST:
            return
        named_matrix = self.species_constraint._makeSpeciesConstraintMatrix()
        self.assertTrue(isinstance(named_matrix, NamedMatrix))
        df = named_matrix.dataframe
        self.assertGreater(len(df), 0)

    def testMakeSpeciesConstraintMatrixScale(self):
        if IGNORE_TEST:
            return
        for _ in range(10):
            size = 20
            network = Network.makeRandomNetworkByReactionType(size, size)
            uni_result =  re.findall(": S. ->", str(network))
            num_uni_re = len(list(uni_result))
            uni_result =  re.findall(": S.. ->", str(network))
            num_uni_re += len(list(uni_result))
            species_constraint = SpeciesConstraint(network.reactant_nmat, network.product_nmat)
            named_matrix = species_constraint._makeSpeciesReactionConstraintMatrix()
            self.assertTrue(isinstance(named_matrix, NamedMatrix))
            df = named_matrix.dataframe
            uni_names = ['r_uni-null', 'r_uni-uni', 'r_uni-bi', 'r_uni-multi']
            num_uni_nmat = df[uni_names].sum().sum()
            self.assertTrue(np.isclose(num_uni_re, num_uni_nmat))

    def testMakeAutocatalysisConstraint(self):
        if IGNORE_TEST:
            return
        for _ in range(30):
            size = 20
            network = Network.makeRandomNetworkByReactionType(size, size)
            autocatalysis_arr = np.zeros(size)
            for stg in [" ", "\n", "$"]:
                autocatalysis_arr +=  np.array([len(re.findall("S" + str(n)
                      + " .*->.*S" + str(n) + stg, str(network)))
                      for n in range(size)])
            valid_idxs = ["S" + str(n) in str(network) for n in range(size)]
            autocatalysis_arr = autocatalysis_arr[valid_idxs]
            species_constraint = SpeciesConstraint(network.reactant_nmat, network.product_nmat)
            named_matrix = species_constraint._makeAutocatalysisConstraint()
            diff = np.abs(np.sum(autocatalysis_arr - named_matrix.values.flatten()))
            if diff != 0:
                import pdb; pdb.set_trace()
            self.assertEqual(diff, 0)

    def testSpeciesConstraintMatrix(self):
        if IGNORE_TEST:
            return
        for _ in range(30):
            size = 20
            network = Network.makeRandomNetworkByReactionType(size, size)
            species_constraint = SpeciesConstraint(network.reactant_nmat, network.product_nmat)
            named_matrix = species_constraint._makeAutocatalysisConstraint()
            self.assertTrue(isinstance(named_matrix, NamedMatrix))
            df = named_matrix.dataframe
            self.assertGreater(len(df), 0)



if __name__ == '__main__':
    unittest.main()