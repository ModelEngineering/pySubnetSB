from pySubnetSB.constraint import Constraint, ReactionClassification, CompatibilityCollection    # type: ignore
from pySubnetSB.named_matrix import NamedMatrix   # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore
from pySubnetSB.constraint_option_collection import ReactionConstraintOptionCollection  # type: ignore

import itertools
import numpy as np
from scipy.special import factorial  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
reactant_arr = np.array([[1, 0], [0, 1], [0, 0]]) # Must have 3 rows to be consistent with DummyConstraint
product_arr = np.array([[0, 1], [1, 0], [0, 0]])
NUM_SPECIES, NUM_REACTION = reactant_arr.shape
SPECIES_NAMES = ["S" + str(i) for i in range(NUM_SPECIES)]
REACTION_NAMES = ["J" + str(i) for i in range(NUM_REACTION)]
REACTANT_NMAT = NamedMatrix(reactant_arr,  row_names=SPECIES_NAMES, column_names=REACTION_NAMES)
PRODUCT_NMAT = NamedMatrix(product_arr,  row_names=SPECIES_NAMES, column_names=REACTION_NAMES)
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
class DummyConstraint(Constraint):
    SPECIES_NAMES = ["A", "B", "C"]
    NUM_ROW = len(SPECIES_NAMES)
    arr = np.random.randint(0, 2, (NUM_ROW, NUM_ROW))
    ONE_STEP_NMAT = NamedMatrix(arr, row_names='rows', column_names='columns')

    
    def __init__(self, reactant_nmat:NamedMatrix, product_nmat:NamedMatrix):
        self.species_names = self.SPECIES_NAMES
        self._num_row = self.NUM_ROW
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat)
        #
        self._categorical_nmat = NamedMatrix(np.array([[0, 1], [1, 0], [1, 1]]),
                row_names=self.species_names, column_names=REACTION_NAMES)
        self._enumerated_nmat = NamedMatrix(np.array([[0, 10], [10, 0], [10, 10]]),
                row_names=self.species_names, column_names=REACTION_NAMES)
        
    def scale(self, scale:int)->'DummyConstraint':
        num_row = scale*len(self.species_names)
        reactant_nmat = NamedMatrix.vstack([self.reactant_nmat]*scale, is_rename=True)
        product_nmat = NamedMatrix.vstack([self.product_nmat]*scale, is_rename=True)
        constraint = DummyConstraint(reactant_nmat, product_nmat)
        equality_arr = np.vstack([self.equality_nmat.values]*scale)
        inequality_arr = np.vstack([self.numerical_inequality_nmat.values]*scale)
        constraint._categorical_nmat = NamedMatrix(equality_arr, row_names=np.array(range(num_row)),
                column_names=REACTION_NAMES)
        constraint._enumerated_nmat = NamedMatrix(inequality_arr, row_names=np.array(range(num_row)),
                column_names=REACTION_NAMES)
        return constraint

    @property
    def categorical_nmat(self)->NamedMatrix:
        return self._categorical_nmat
    
    @property
    def numerical_enumerated_nmat(self)->NamedMatrix:
        return self._enumerated_nmat

    @property
    def bitwise_enumerated_nmat(self)->NamedMatrix:
        return self._enumerated_nmat
    
    @property
    def one_step_nmat(self)->NamedMatrix:
        arr = np.matmul(self.product_nmat.values, self.reactant_nmat.values.T)
        row_names = [str(n) for n in range(arr.shape[0])]
        return NamedMatrix(arr, row_names=row_names, column_names=row_names)

    @classmethod 
    def makeDummyConstraint(cls, num_species:int=3, num_reaction:int=3):
        for _ in range(100):
            network = Network.makeRandomNetworkByReactionType(num_reaction, num_species)
            if (num_species != network.num_species) or (num_reaction != network.num_reaction):
                continue
            break
        else:
            raise RuntimeError("Failed to make a random network")
        return DummyConstraint(network.reactant_nmat, network.product_nmat)
    
    @property
    def row_names(self)->np.ndarray:
        return np.array(SPECIES_NAMES)


#############################
class ScalableDummyConstraint(Constraint):
    
    def __init__(self, reference_size):
        reactant_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (reference_size, reference_size))))
        product_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (reference_size, reference_size))))
        super().__init__(reactant_nmat=reactant_nmat, product_nmat=product_nmat)
        #
        self.num_species = len(reactant_nmat.row_names)
        num_column = 5
        self._categorical_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (self.num_species, num_column))))
        self._enumerated_nmat = NamedMatrix(np.array(np.random.randint(0, 2, (self.num_species, num_column))))
        
    def scale(self, scale:int)->'ScalableDummyConstraint':
        reactant_nmat = NamedMatrix.vstack([self.reactant_nmat]*scale, is_rename=True)
        product_nmat = NamedMatrix.vstack([self.product_nmat]*scale, is_rename=True)
        constraint = DummyConstraint(reactant_nmat, product_nmat)
        equality_arr = np.vstack([self.equality_nmat.values]*scale)
        inequality_arr = np.vstack([self.numerical_inequality_nmat.values]*scale)
        num_row = equality_arr.shape[0]
        constraint._categorical_nmat = NamedMatrix(equality_arr, row_names=np.array(range(num_row)))
        constraint._enumerated_nmat = NamedMatrix(inequality_arr, row_names=np.array(range(num_row)))
        return constraint

    @property
    def categorical_nmat(self)->NamedMatrix:
        return self._categorical_nmat
    
    @property
    def numerical_enumerated_nmat(self)->NamedMatrix:
        return self._enumerated_nmat
    
    @property
    def bitwise_enumerated_nmat(self)->NamedMatrix:
        return self._enumerated_nmat
    
    @property
    def one_step_nmat(self)->NamedMatrix:
        arr = np.matmul(self.product_nmat.values, self.reactant_nmat.values.T)
        return NamedMatrix(arr,
                row_names='rows', column_names='columns')
    

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

    def testMakeReactionClassificationMatrix(self):
        if IGNORE_TEST:
            return
        reaction_classifications = [ReactionClassification(num_reactant=1, num_product=2),
              ReactionClassification(num_reactant=2, num_product=1)]
        reaction_names = ["J1", "J2"]
        nmat = ReactionClassification.makeReactionClassificationMatrix(reaction_names, reaction_classifications)
        df = nmat.dataframe
        self.assertTrue(df.loc["J1", "uni-bi"] == 1)
        self.assertTrue(df.loc["J2", "bi-uni"] == 1)
        for _, row in df.iterrows():
            self.assertEqual(np.sum(row), 1)


#############################
class TestCompatibilityCollection(unittest.TestCase):

    def setUp(self) -> None:
        self.collection = CompatibilityCollection(2, 3)

    def testNumPermutation(self):
        if IGNORE_TEST:
            return
        for size in np.random.randint(2, 100, 10):
            collection = CompatibilityCollection(size, size)
            [collection.add(i-1, range(i)) for i in range(1, size+1)]
            self.assertLessEqual(collection.log10_num_assignment, np.log10(factorial(size)))

    def testPrune(self):
        if IGNORE_TEST:
            return
        log10_max_permutation = 4.0
        for size in np.random.randint(5, 30, 100):
            collection = CompatibilityCollection(size, size)
            [collection.add(i-1, range(i)) for i in range(1, size+1)]
            new_collection, is_changed = collection.prune(log10_max_permutation)
            result = (collection.log10_num_assignment <= log10_max_permutation) and (not is_changed)
            result = result or (collection.log10_num_assignment > log10_max_permutation) and is_changed
            self.assertTrue(result)
            if not is_changed:
                self.assertEqual(new_collection, collection)
            else:
                self.assertNotEqual(new_collection, collection)

    def testExpand(self):
        if IGNORE_TEST:
            return
        size = 3
        collection = CompatibilityCollection(size, size)
        [collection.add(i-1, range(i)) for i in range(1, size+1)]
        arr, _ = collection.expand()
        self.assertLessEqual(arr.shape[0], factorial(3))
        self.assertEqual(arr.shape[1], 3)


#############################
class TestConstraint(unittest.TestCase):

    def setUp(self):
        self.constraint = DummyConstraint(reactant_nmat=REACTANT_NMAT, product_nmat=PRODUCT_NMAT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(REACTANT_NMAT, self.constraint.reactant_nmat)
        self.assertEqual(PRODUCT_NMAT, self.constraint.product_nmat)

    def testEq(self):
        if IGNORE_TEST:
            return
        constraint = DummyConstraint(reactant_nmat=REACTANT_NMAT, product_nmat=PRODUCT_NMAT)
        self.assertTrue(self.constraint == constraint)
        #
        constraint = DummyConstraint(reactant_nmat=PRODUCT_NMAT, product_nmat=REACTANT_NMAT)
        self.assertFalse(self.constraint == constraint)

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
        result, _ = self.constraint.calculateBooleanCompatibilityVector(self.constraint.equality_nmat,
              self.constraint.equality_nmat, is_equality=True)
        num_true = np.sum(result)
        self.assertEqual(num_true, self.constraint.equality_nmat.num_row)
        #
        result, _ = self.constraint.calculateBooleanCompatibilityVector(self.constraint.numerical_inequality_nmat,
              self.constraint.equality_nmat, is_equality=True)
        num_true = np.sum(result)
        self.assertEqual(num_true, 0)
        #
        result, _ = self.constraint.calculateBooleanCompatibilityVector(self.constraint.equality_nmat,
              self.constraint.numerical_inequality_nmat, is_equality=False)
        num_true = np.sum(result)
        self.assertEqual(num_true, 5)

    def testMakeCompatibilityCollection(self):
        if IGNORE_TEST:
            return
        compatibility_collection = self.constraint.makeCompatibilityCollection(self.constraint).compatibility_collection
        self.assertTrue(np.isclose(compatibility_collection.log10_num_assignment, 0))
    
    def testMakeCompatibilityCollectionScale(self):
        if IGNORE_TEST:
            return
        scale = 10  # scaling factor
        constraint = ScalableDummyConstraint(50)
        new_constraint = constraint.scale(scale)
        compatibility_collection = constraint.makeCompatibilityCollection(new_constraint).compatibility_collection
        trues = [len(lst) >= scale
                   for lst in compatibility_collection.compatibilities]
        self.assertTrue(all(trues))

    def testCalculateLog10UnconstrainedPermutation(self):
        if IGNORE_TEST:
            return
        for size in range(3, 20):
            log10_permutation = self.constraint.calculateLog10UnconstrainedPermutation(size, size)
            self.assertTrue(np.isclose(log10_permutation, 2*np.log10(factorial(size))))

    def testExpandReductionInSize(self):
        if IGNORE_TEST:
            return
        fill_size = 2
        for size in range(3, 20):
            network = Network.makeRandomNetworkByReactionType(size, size)
            large_network = network.fill(num_fill_reaction=fill_size*size, num_fill_species=fill_size*size)
            large_constraint = ReactionConstraint(large_network.reactant_nmat, large_network.product_nmat)
            constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat)
            compatibility_collection = constraint.makeCompatibilityCollection(large_constraint).compatibility_collection
            result = compatibility_collection.expand()
            self.assertFalse(result[1])
            self.assertEqual(result[0].shape[1], size)


if __name__ == '__main__':
    unittest.main(failfast=True)