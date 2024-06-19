from sirn.stoichometry import Stoichiometry  # type: ignore
import sirn.constants as cn  # type: ignore

import os
import numpy as np
import tellurium as te # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_MODELS = 10
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")

#############################
# Tests
#############################
class TestStoichiometryMatrices(unittest.TestCase):

    def testSimple(self):
        if IGNORE_TEST:
            return
        antimony_str = '''
        model test
            J0: S1 -> S2; k1*S1;
            k1 = 0.1;
            S1 = 10;
        end
        '''
        stoichiometry = Stoichiometry(antimony_str)
        self.assertTrue(np.all(stoichiometry.reactant_mat == np.array([[1], [0]])))
        self.assertTrue(np.all(stoichiometry.product_mat == np.array([[0], [1]])))
        self.assertTrue(np.all(stoichiometry.stoichiometry_mat == np.array([[-1], [1]])))

    def testOnModels(self):
        if IGNORE_TEST:
            return
        ffiles = os.listdir(MODEL_DIR)
        for ffile in ffiles:
            path = os.path.join(MODEL_DIR, ffile)
            rr = te.loada(path)
            antimony_str = rr.getAntimony()
            smat = rr.getFullStoichiometryMatrix()
            stoichiometry = Stoichiometry(antimony_str)
            self.assertTrue(np.all(stoichiometry.stoichiometry_mat == smat))
            self.assertTrue(all([x == y for x, y in zip(stoichiometry.species_names, smat.rownames)]))
            self.assertTrue(all([x == y for x, y in zip(stoichiometry.reaction_names, smat.colnames)]))


if __name__ == '__main__':
    unittest.main()