from sirn.assignment_pair import AssignmentPair  # type: ignore

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

        

if __name__ == '__main__':
    unittest.main()