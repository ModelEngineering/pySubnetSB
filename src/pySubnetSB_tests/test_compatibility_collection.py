from pySubnetSB.compatibility_collection import CompatibilityCollection  # type: ignore
from pySubnetSB.named_matrix import NamedMatrix   # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore

import numpy as np
from scipy.special import factorial  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False


class TestCompatibilityCollection(unittest.TestCase):

    def setUp(self) -> None:
        self.collection = CompatibilityCollection(2, 3)

    def testNumAssignment(self):
        if IGNORE_TEST:
            return
        for size in np.random.randint(2, 7, 100):
            collection = CompatibilityCollection(size, size)
            [collection.add(i-1, np.random.randint(0, size, size)) for i in range(1, size+1)]
            expanded_collection, _ = collection.expand()
            if expanded_collection.shape[0] > 0:
                log10_expanded_collection = max(0, np.log10(expanded_collection.shape[0]))
            else:
                log10_expanded_collection = 0
            self.assertTrue(np.abs(collection.log10_num_assignment - log10_expanded_collection) < 2)

    def testPrune(self):
        if IGNORE_TEST:
            return
        log10_max_assignment = 4.0
        for size in np.random.randint(5, 30, 100):
            collection = CompatibilityCollection(size, size)
            [collection.add(i-1, range(i)) for i in range(1, size+1)]
            new_collection, is_changed = collection.prune(log10_max_assignment)
            result = (collection.log10_num_assignment <= log10_max_assignment) and (not is_changed)
            result = result or (collection.log10_num_assignment > log10_max_assignment) and is_changed
            self.assertTrue(result)
            if not is_changed:
                self.assertEqual(new_collection, collection)
            else:
                self.assertNotEqual(new_collection, collection)

    def testExpand(self):
        if IGNORE_TEST:
            return
        size = 5
        collection = CompatibilityCollection(size, size)
        [collection.add(i-1, range(i)) for i in range(1, size+1)]
        arr, is_truncated = collection.expand()
        self.assertFalse(is_truncated)
        self.assertLessEqual(arr.shape[0], 1)
        self.assertEqual(arr.shape[1], size)

    def testExpandTruncated(self):
        if IGNORE_TEST:
            return
        size = 40
        collection = CompatibilityCollection(size, size)
        [collection.add(i-1, range(i)) for i in range(1, size+1)]
        arr, is_truncated = collection.expand(max_num_assignment=10)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertTrue(is_truncated)

    def testExpandReductionInSize(self):
        if IGNORE_TEST:
            return
        fill_size = 5
        for size in range(5, 10):
            network = Network.makeRandomNetworkByReactionType(size, size)
            large_network = network.fill(num_fill_reaction=fill_size*size, num_fill_species=fill_size*size)
            large_constraint = ReactionConstraint(large_network.reactant_nmat, large_network.product_nmat)
            constraint = ReactionConstraint(network.reactant_nmat, network.product_nmat)
            compatibility_collection = constraint.makeCompatibilityCollection(
                  large_constraint).compatibility_collection
            arr, _ = compatibility_collection.expand()
            for row in arr:
                for idx in range(len(row)):
                    expected_name = "J" + str(idx)
                    if expected_name in large_network.reaction_names[row]:
                        break
                else:
                    self.assertTrue(False)

    def testBug(self):
        if IGNORE_TEST:
            return
        list_of_lists = [[3, 11, 16, 17], [2, 8, 14, 15, 17], [1], [2, 4, 5, 7, 12], [17],
                                    [2, 4, 5, 7, 12], [0, 8], [6, 13], [6, 13], [3, 11, 16, 17], [3, 11, 16, 17]]
        collection = CompatibilityCollection.makeFromListOfLists(list_of_lists)
        arr, _ = collection.expand()
        self.assertEqual(arr.shape[1], len(list_of_lists))


if __name__ == '__main__':
    unittest.main(failfast=True)