from sirn.species_constraint import SpeciesConstraint # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore

import numpy as np
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
REACTANT_MATRIX = NamedMatrix(np.array([[0, 1], [1, 0], [0, 0]]))
PRODUCT_MATRIX = NamedMatrix(np.array([[0, 1], [1, 0], [0, 0]]))


#############################
# Tests
#############################
class TestSpeciesConstraint(unittest.TestCase):

    def setUp(self):
        self.species_constraint = SpeciesConstraint(REACTANT_MATRIX.copy(), PRODUCT_MATRIX.copy()) 



if __name__ == '__main__':
    unittest.main()