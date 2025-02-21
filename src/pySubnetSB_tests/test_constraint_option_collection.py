from pySubnetSB.constraint_option_collection import ReactionConstraintOptionCollection # type: ignore
from pySubnetSB.constraint_option_collection import SpeciesConstraintOptionCollection  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint  # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import numpy as np
from scipy.special import factorial  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
MODEL1 = """
J0: S0 -> S1; k0*S0
J1: S1 -> S0; k1*S1
k0 = 0.1; k1 = 0.2;
S0 = 10; S1 = 20
"""
NETWORK1 = Network.makeFromAntimonyStr(MODEL1)
MODEL2 = """
J0: S0 -> S1; k0*S0
J1: S1 -> S1; k1*S1
J2: S0 + S1 -> S1; k2*S0*S1
k0 = 0.1; k1 = 0.2; k2 = 0.3
S0 = 10; S1 = 20; S2 = 30
"""
NETWORK2 = Network.makeFromAntimonyStr(MODEL2)


#############################
# Tests
#############################
class TestConstraintOptionCollection(unittest.TestCase):

    def setUp(self):
        if np.random.rand() > 0.5:
            self.option_collection = ReactionConstraintOptionCollection()
        else:
            self.option_collection = SpeciesConstraintOptionCollection()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(all(self.option_collection.__dict__.values()))

    def testMakeFromShortName(self):
        if IGNORE_TEST:
            return
        collection_short_name = self.option_collection.collection_short_name
        new_option = self.option_collection.makeFromCollectionShortName(collection_short_name)
        self.assertEqual(self.option_collection, new_option)

    def test_sort_names(self):
        #if IGNORE_TEST:
        #    return
        short_name = self.option_collection.collection_short_name
        self.assertTrue("+" in short_name)
        #
        self.option_collection.setAllFalse()
        short_name = self.option_collection.collection_short_name
        self.assertEqual(short_name, cn.NONE)

    def testGetTrueNames(self):
        if IGNORE_TEST:
            return
        names = self.option_collection.getTrueNames()
        num_true = np.sum([self.option_collection.__dict__[k]
              for k, v in self.option_collection.__dict__.items() if isinstance(v, bool)])
        self.assertEqual(len(names), num_true)
        #
        self.option_collection.setAllFalse()
        names = self.option_collection.getTrueNames()
        self.assertEqual(len(names), 0)

    def testIterator(self):
        if IGNORE_TEST:
            return
        options = list(self.option_collection.iterator())
        self.assertEqual(len(options), 2**len(self.option_collection.option_names))
        trues = [isinstance(opt, self.option_collection.__class__) for opt in options]
        self.assertTrue(all(trues))

    def testConstraintOptionEffect(self):
        # Test if add constraints reduces number of assignments
        if IGNORE_TEST:
            return
        all_reference_constraint = SpeciesConstraint(NETWORK1.reactant_nmat, NETWORK1.product_nmat)
        all_target_constraint = SpeciesConstraint(NETWORK2.reactant_nmat, NETWORK2.product_nmat)
        all_compatibility_collection = all_reference_constraint.makeCompatibilityCollection(
            all_target_constraint).compatibility_collection
        constraint_options = SpeciesConstraintOptionCollection()
        constraint_options.setAllFalse()
        no_reference_constraint = SpeciesConstraint(NETWORK1.reactant_nmat, NETWORK1.product_nmat,
              species_constraint_option_collection=constraint_options)
        no_constraint = SpeciesConstraint(NETWORK2.reactant_nmat, NETWORK2.product_nmat,
              species_constraint_option_collection=constraint_options)
        no_compatibility_collection = no_reference_constraint.makeCompatibilityCollection(
            no_constraint).compatibility_collection
        # The compatibility set should be larger with no constraints
        trues = [len(x) > len(y) for x, y in zip(no_compatibility_collection.compatibilities,
                                                 all_compatibility_collection.compatibilities)]
        self.assertTrue(all(trues))


if __name__ == '__main__':
    unittest.main(failfast=True)