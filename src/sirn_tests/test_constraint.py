from sirn.constraint import Constraint, ReactionClassification      # type: ignore
from sirn.named_matrix import NamedMatrix   # type: ignore

import itertools
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
REACTION_NAMES = ["J1", "J2"]
SPECIES_NAMES = ["A", "B"]
reactant_arr = np.array([[1, 0], [0, 1]])
product_arr = np.array([[0, 1], [1, 0]])
REACTANT_NMAT = NamedMatrix(reactant_arr,  row_names=SPECIES_NAMES, column_names=REACTION_NAMES)
PRODUCT_NMAT = NamedMatrix(product_arr,  row_names=SPECIES_NAMES, column_names=REACTION_NAMES)



#############################
class TestReactionClassification(unittest.TestCase):

    def setUp(self):
        self.reaction_classification = ReactionClassification(num_reactant=1, num_product=2)
    
    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(str(self.reaction_classification), "uni-bi")

    def testMany(self):
        if IGNORE_TEST:
            return
        iter =  itertools.product(range(4), repeat=2)
        for num_reactant, num_product in iter:
            reaction_classification = ReactionClassification(num_reactant=num_reactant,
                  num_product=num_product)
            if (num_reactant == 0) or (num_product == 0):
                self.assertTrue("null" in str(reaction_classification))
            if (num_reactant == 1) or (num_product == 1):
                self.assertTrue("uni" in str(reaction_classification))
            if (num_reactant == 2) or (num_product == 2):
                self.assertTrue("bi" in str(reaction_classification))
            if (num_reactant == 3) or (num_product == 3):
                self.assertTrue("multi" in str(reaction_classification))


#############################
class TestConstraint(unittest.TestCase):

    def setUp(self):
        self.constraint = Constraint(reactant_nmat=REACTANT_NMAT, product_nmat=PRODUCT_NMAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(REACTANT_NMAT, self.constraint.reactant_nmat)
        self.assertEqual(PRODUCT_NMAT, self.constraint.product_nmat)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        constraint = self.constraint.copy()
        self.assertTrue(self.constraint == constraint.copy())
        #
        self.constraint.reactant_nmat.values[0,0] = 100
        self.assertFalse(self.constraint == constraint.copy())

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        serialization_str = self.constraint.serialize()
        constraint = Constraint.deserialize(serialization_str)
        self.assertEqual(self.constraint, constraint)
    
    def testClassifyReactions(self):
        if IGNORE_TEST:
            return
        reaction_classifications = self.constraint.classifyReactions()
        self.assertTrue(all([isinstance(rc, ReactionClassification) for rc in reaction_classifications]))


if __name__ == '__main__':
    unittest.main()