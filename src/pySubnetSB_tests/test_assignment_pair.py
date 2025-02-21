from pySubnetSB.assignment_pair import AssignmentPair  # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import copy
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
SPECIES_ASSIGNMENT = np.array(range(3))
REACTION_ASSIGNMENT = np.array(range(4))


#############################
# Tests
#############################
class TestAssignmentPair(unittest.TestCase):

    def setUp(self):
        self.assignment_pair = AssignmentPair(species_assignment=SPECIES_ASSIGNMENT, reaction_assignment=REACTION_ASSIGNMENT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(np.all(self.assignment_pair.species_assignment==SPECIES_ASSIGNMENT))
        self.assertTrue(np.all(self.assignment_pair.reaction_assignment==REACTION_ASSIGNMENT))

    def testEq(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.assignment_pair, self.assignment_pair)
        assignment_pair = AssignmentPair(species_assignment=REACTION_ASSIGNMENT, reaction_assignment=REACTION_ASSIGNMENT)
        self.assertNotEqual(self.assignment_pair, assignment_pair)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        serialization_str = self.assignment_pair.serialize()
        assignment_pair = self.assignment_pair.deserialize(serialization_str)
        self.assertEqual(self.assignment_pair, assignment_pair)

    def testInvert(self):
        if IGNORE_TEST:
            return
        def check(perm, inv_perm):
            identity_perm = range(len(perm))
            self.assertTrue(np.all(perm[inv_perm] == identity_perm))
        #####
        assignment_pair = self.assignment_pair.invert()
        check(self.assignment_pair.species_assignment, assignment_pair.species_assignment)
        check(self.assignment_pair.reaction_assignment, assignment_pair.reaction_assignment)

    def testMakeNameDct(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(ValueError):
            self.assignment_pair.makeNameDct()
        #
        reference_species_names = ["species_" + str(n) for n in range(len(SPECIES_ASSIGNMENT))]
        reference_reaction_names = ["reaction_" + str(n) for n in range(len(REACTION_ASSIGNMENT))]
        target_species_names = ["species_" + str(n) for n in range(len(SPECIES_ASSIGNMENT)+1)]
        target_reaction_names = ["reaction_" + str(n) for n in range(len(REACTION_ASSIGNMENT)+1)]
        assignment_pair = AssignmentPair(species_assignment=SPECIES_ASSIGNMENT, reaction_assignment=REACTION_ASSIGNMENT,
              reference_reaction_names=reference_reaction_names, reference_species_names=reference_species_names,
              target_reaction_names=target_reaction_names, target_species_names=target_species_names)
        dct = assignment_pair.makeNameDct()
        self.assertTrue(cn.SPECIES_NAMES in dct)
        self.assertTrue(cn.REACTION_NAMES in dct)
        for superkey in [cn.SPECIES_NAMES, cn.REACTION_NAMES]:
            for key, value in dct[superkey].items():
                self.assertEqual(key, value)

        

if __name__ == '__main__':
    unittest.main()