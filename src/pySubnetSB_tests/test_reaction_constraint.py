from pySubnetSB.constraint import NULL_NMAT  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint # type: ignore
from pySubnetSB.named_matrix import NamedMatrix # type: ignore
from pySubnetSB.network import Network # type: ignore

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
        self.assertEqual(self.constraint._numerical_categorical_nmat, NULL_NMAT)
        self.assertEqual(self.constraint._numerical_enumerated_nmat, NULL_NMAT)

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
    
    def testMakeSuccessorPredecessorConstraintMatrix(self):
        if IGNORE_TEST:
            return
        frac_duplicates = []
        for _ in range(100):
            size = 5
            network = Network.makeRandomNetworkByReactionType(size, size)
            constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat)
            named_matrix = constraint.makeSuccessorPredecessorConstraintMatrix()
            num_unique = len(np.unique(named_matrix.values, axis=0))
            frac_duplicate = 1 - num_unique / len(named_matrix.values)
            frac_duplicates.append(frac_duplicate)
            sum_arr = named_matrix.values.sum(axis=1)
            self.assertTrue(isinstance(named_matrix, NamedMatrix))
            self.assertTrue(np.all(sum_arr >= 0))
        #print(np.mean(frac_duplicates))

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

    def testcategoricalAndEnumeratedConstraints(self):
        if IGNORE_TEST:
            return
        for _ in range(4):
            self.constraint.setSubset(True)
            self.assertTrue(self.constraint.equality_nmat is not NULL_NMAT)
            self.assertTrue(self.constraint.numerical_inequality_nmat is not NULL_NMAT)
            #
            self.constraint.setSubset(False)
            self.assertTrue(self.constraint.equality_nmat is not NULL_NMAT)
            self.assertTrue(self.constraint.numerical_inequality_nmat is NULL_NMAT)

    def testmakeCompatibilityCollection(self):
        if IGNORE_TEST:
            return
        num_permutations = []
        for _ in range(100):
            reference_size = 18
            filler_size = 3*reference_size
            network = Network.makeRandomNetworkByReactionType(reference_size, reference_size)
            big_network = network.fill(num_fill_reaction=filler_size, num_fill_species=filler_size)
            # Not doing initialization
            reaction_constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat,
                                                   is_subnet=True)
            big_reaction_constraint = ReactionConstraint(big_network.reactant_nmat, big_network.product_nmat,
                                                       is_subnet=True)
            compatibility_collection = reaction_constraint.makeCompatibilityCollection(
                  big_reaction_constraint).compatibility_collection
            name_arr = np.array(big_reaction_constraint.reactant_nmat.column_names)
            for i, arr in enumerate(compatibility_collection.compatibilities):
                reference_name = "J" + str(i)
                target_names = [name_arr[i] for i in arr]
                self.assertTrue(reference_name in target_names)
            num_permutations.append(compatibility_collection.log10_num_assignment)
        #print(np.mean(num_permutations))

    def testBug(self):
        if IGNORE_TEST:
            return
        small_model = """
        J0: S1 -> S1
        J1: S0 + S1 -> S0 + S1
        J2: S0 -> S0
        """
        network = Network.makeFromAntimonyStr(small_model)
        big_model = """
        J5: S2 -> S2 + S4
  J0: S1 -> S1
  J4: S3 -> S3
  J1: S1 + S0 -> S1 + S0
  J3:  -> S4
  J2: S0 -> S0
        """
        big_network = Network.makeFromAntimonyStr(big_model)
        reaction_constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat,
                                                   is_subnet=True)
        big_reaction_constraint = ReactionConstraint(big_network.reactant_nmat, big_network.product_nmat,
                                                       is_subnet=True)
        compatibility_collection = reaction_constraint.makeCompatibilityCollection(
                  big_reaction_constraint).compatibility_collection
        name_arr = np.array(big_reaction_constraint.reactant_nmat.column_names)
        for i, arr in enumerate(compatibility_collection.compatibilities):
                reference_name = "J" + str(i)
                target_names = [name_arr[i] for i in arr]
                self.assertTrue(reference_name in target_names)

    def testBugReactionConstraint(self):
        #if IGNORE_TEST:
        #    return
        reference_model = """
            A -> B; 1
            B -> C; 1
            A = 0
            B = 0
            """
        reference_network = Network.makeFromAntimonyStr(reference_model)
        reference_constraint = ReactionConstraint(reference_network.reactant_nmat, reference_network.product_nmat,
                is_subnet=True)
        target_model = """
            S1 -> S2; 2
            S2 -> S3; 2
            S3 -> S4; 2
            A = 0
            B = 0
            C = 0
            """
        target_network = Network.makeFromAntimonyStr(target_model)
        target_constraint = ReactionConstraint(target_network.reactant_nmat, target_network.product_nmat,
                is_subnet=True)
        compatibility_collection = reference_constraint.makeCompatibilityCollection(
              target_constraint).compatibility_collection
        trues = [len(v) > 0 for v in compatibility_collection.compatibilities]
        self.assertTrue(all(trues))


if __name__ == '__main__':
    unittest.main()