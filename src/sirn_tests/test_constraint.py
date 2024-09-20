from sirn.constraint import Constraint, ReactionClassification, CompatibilityCollection    # type: ignore
from sirn.named_matrix import NamedMatrix   # type: ignore

import itertools
import numpy as np
from scipy.special import factorial
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
class DummyConstraint(Constraint):
    
    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix):
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat)
        #
        self.species_names = ["A", "B", "C"]
        self._equality_nmat = NamedMatrix(np.array([[0, 1], [1, 0], [1, 1]]),
                row_names=self.species_names, column_names=REACTION_NAMES)
        self._inequality_nmat = NamedMatrix(np.array([[0, 10], [10, 0], [10, 10]]),
                row_names=self.species_names, column_names=REACTION_NAMES)
        
    def scale(self, scale:int)->'DummyConstraint':
        num_row = scale*len(self.species_names)
        reactant_nmat = NamedMatrix.vstack([self.reactant_nmat]*scale, is_rename=True)
        product_nmat = NamedMatrix.vstack([self.product_nmat]*scale, is_rename=True)
        constraint = DummyConstraint(reactant_nmat, product_nmat)
        equality_arr = np.vstack([self.equality_nmat.values]*scale)
        inequality_arr = np.vstack([self.inequality_nmat.values]*scale)
        constraint._equality_nmat = NamedMatrix(equality_arr, row_names=np.array(range(num_row)),
                column_names=REACTION_NAMES)
        constraint._inequality_nmat = NamedMatrix(inequality_arr, row_names=np.array(range(num_row)),
                column_names=REACTION_NAMES)
        return constraint

    
    @property
    def equality_nmat(self)->NamedMatrix:
        return self._equality_nmat
    
    @property
    def inequality_nmat(self)->NamedMatrix:
        return self._inequality_nmat
    


#############################
class ScalableDummyConstraint(Constraint):
    
    def __init__(self, reference_size):
        reactant_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (reference_size, reference_size))))
        product_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (reference_size, reference_size))))
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat)
        #
        self.num_species = len(reactant_nmat.row_names)
        num_column = 5
        self._equality_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (self.num_species, num_column))))
        self._inequality_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (self.num_species, num_column))))
        
    def scale(self, scale:int)->'ScalableDummyConstraint':
        reactant_nmat = NamedMatrix.vstack([self.reactant_nmat]*scale, is_rename=True)
        product_nmat = NamedMatrix.vstack([self.product_nmat]*scale, is_rename=True)
        constraint = DummyConstraint(reactant_nmat, product_nmat)
        equality_arr = np.vstack([self.equality_nmat.values]*scale)
        inequality_arr = np.vstack([self.inequality_nmat.values]*scale)
        constraint._equality_nmat = NamedMatrix(equality_arr, row_names=np.array(range(self.num_row)))
        constraint._inequality_nmat = NamedMatrix(inequality_arr, row_names=np.array(range(self.num_row)))
        return constraint

    
    @property
    def equality_nmat(self)->NamedMatrix:
        return self._equality_nmat
    
    @property
    def inequality_nmat(self)->NamedMatrix:
        return self._inequality_nmat


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
class TestCompatibilityCollection(unittest.TestCase):

    def setUp(self) -> None:
        self.collection = CompatibilityCollection(2, 3)

    def testNumPermutation(self):
        if IGNORE_TEST:
            return
        for size in np.random.randint(2, 20, 1000):
            collection = CompatibilityCollection(size, size)
            [collection.add(i-1, range(i)) for i in range(1, size+1)]
            self.assertTrue(np.isclose(collection.num_permutation, factorial(size)))

    def testPrune(self):
        if IGNORE_TEST:
            return
        max_permutation = 1000
        for size in np.random.randint(5, 20, 100):
            collection = CompatibilityCollection(size, size)
            [collection.add(i-1, range(i)) for i in range(1, size+1)]
            new_collection = collection.prune(max_permutation)
            self.assertTrue(new_collection.num_permutation <= max_permutation)


#############################
class TestConstraint(unittest.TestCase):

    def setUp(self):
        self.constraint = DummyConstraint(reactant_nmat=REACTANT_NMAT, product_nmat=PRODUCT_NMAT)

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
        constraint = DummyConstraint.deserialize(serialization_str)
        self.assertEqual(self.constraint, constraint)
    
    def testClassifyReactions(self):
        if IGNORE_TEST:
            return
        reaction_classifications = self.constraint.classifyReactions()
        self.assertTrue(all([isinstance(rc, ReactionClassification) for rc in reaction_classifications]))

    def testCalculateCompatibility(self):
        if IGNORE_TEST:
            return
        result = self.constraint.calculateBooleanCompatibilityVector(self.constraint.equality_nmat,
              self.constraint.equality_nmat, is_equality=True)
        num_true = np.sum(result)
        self.assertEqual(num_true, self.constraint.equality_nmat.num_row)
        #
        result = self.constraint.calculateBooleanCompatibilityVector(self.constraint.inequality_nmat,
              self.constraint.equality_nmat, is_equality=True)
        num_true = np.sum(result)
        self.assertEqual(num_true, 0)
        #
        result = self.constraint.calculateBooleanCompatibilityVector(self.constraint.equality_nmat,
              self.constraint.inequality_nmat, is_equality=False)
        num_true = np.sum(result)
        self.assertEqual(num_true, 5)

    def testMakeCompatibilityCollection(self):
        if IGNORE_TEST:
            return
        compatibility_collection = self.constraint.makeCompatibilityCollection(self.constraint)
        self.assertEqual(compatibility_collection.num_permutation, 1)
    
    def testMakeCompatibilityCollectionScale(self):
        if IGNORE_TEST:
            return
        scale = 5000  # scaling factor
        constraint = ScalableDummyConstraint(50)
        new_constraint = constraint.scale(scale)
        compatibility_collection = constraint.makeCompatibilityCollection(new_constraint)
        trues = [len(lst) >= scale
                   for lst in compatibility_collection.compatibilities]
        self.assertTrue(all(trues))


if __name__ == '__main__':
    unittest.main()