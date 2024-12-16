from pySubnetSB.stoichometry import Stoichiometry  # type: ignore
import pySubnetSB.constants as cn  # type: ignore

import os
import numpy as np
import tellurium as te # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_MODELS = 10
MODEL_DIR = os.path.join(cn.TEST_DIR, "oscillators")
MODEL1 = """
model test
            J0: S1 -> S2; k1*S1;
            k1 = 0.1;
            S1 = 10;
        end
"""
MODEL2 = """
model test
            J0: $S1 -> S2; k1*S1;
            k1 = 0.1;
            S1 = 10;
        end
"""

#############################
# Tests
#############################
class TestStoichiometryMatrices(unittest.TestCase):

    def testSimple(self):
        if IGNORE_TEST:
            return
        antimony_str = MODEL2
        stoichiometry = Stoichiometry(antimony_str)
        self.assertTrue(np.all(stoichiometry.reactant_mat == np.array([[0], [1]])))
        self.assertTrue(np.all(stoichiometry.product_mat == np.array([[1], [0]])))
        self.assertTrue(np.all(stoichiometry.standard_mat == np.array([[1], [-1]])))

    def testOnModels(self):
        if IGNORE_TEST:
            return
        ffiles = os.listdir(MODEL_DIR)
        for ffile in ffiles:
            path = os.path.join(MODEL_DIR, ffile)
            try:
                rr = te.loada(path)
            except:
                continue
            antimony_str = rr.getAntimony()
            smat = rr.getFullStoichiometryMatrix()
            stoichiometry = Stoichiometry(antimony_str)
            self.assertTrue(np.all(stoichiometry.standard_mat == smat))
            self.assertTrue(all([x == y for x, y in zip(stoichiometry.species_names, smat.rownames)]))
            self.assertTrue(all([x == y for x, y in zip(stoichiometry.reaction_names, smat.colnames)]))

    def testWithFile(self):
        if IGNORE_TEST:
            return
        path = os.path.join(MODEL_DIR, "bestmodel_9eGqe98Sp3ry")
        rr = te.loada(path)
        antimony_str = rr.getAntimony()
        smat = rr.getFullStoichiometryMatrix()
        stoichiometry = Stoichiometry(antimony_str)
        self.assertTrue(np.all(stoichiometry.standard_mat == smat))
        self.assertTrue(all([x == y for x, y in zip(stoichiometry.species_names, smat.rownames)]))
        self.assertTrue(all([x == y for x, y in zip(stoichiometry.reaction_names, smat.colnames)]))

    # FIXME: Finish test
    def testCalculateBug(self):
        if IGNORE_TEST:
            return
        model = """
            R_13: S8 -> S7   ; 1       
            R_6: S3 -> S3 + S9; 1
            R_15: S7 + S9 -> S9; 1
            R_24: S7 -> S7 + S3; 1
            R_25: S3 + S9 -> S9; 1
            R_14: S7 -> 2.0 S7; 1
            R7: S7 + S9 -> S7; 1

            S8=1; S7=1; S3=1; S9=1
        """
        model = """
       R_1: xFinal_10 => xFinal_12; 1
       R_2: xFinal_12 => ; 1
       R_3: xFinal_9 => xFinal_10; 1
       R_4: xFinal_10 => ; 1
       R_5: xFinal_8 => 2 xFinal_9; 1
       R_6: xFinal_8 => xFinal_8 + xFinal_9; 1
       R_7: xFinal_9 => ; 1
       R_8: xFinal_9 => xFinal_8; 1
       R_9: xFinal_7 => 2 xFinal_8; 1
       R_10: xFinal_7 => xFinal_7 + xFinal_8; 1
       R_11: xFinal_8 => 2 xFinal_8; 1
       R_12: xFinal_8 => ; 1
       R_13: xFinal_8 => xFinal_7; 1
       R_14: xFinal_7 => 2 xFinal_7; 1
       R_15: xFinal_6 + xFinal_7 => xFinal_6; 1
       R_16: xFinal_7 => ; 1
       R_17: xFinal_5 => xFinal_6; 1
       R_18: xFinal_6 => ; 1
       R_19: xFinal_4 => xFinal_5; 1
       R_20: xFinal_5 => ; 1
       R_21: xFinal_3 => xFinal_4; 1
       R_22: xFinal_4 => ; 1
       R_23: xFinal_2 => 2 xFinal_3; 1
       R_24: xFinal_2 => xFinal_2 + xFinal_3; 1
       R_25: xFinal_3 => ; 1
       R_26: xFinal_3 => xFinal_2; 1
       R_27: xFinal_1 + xFinal_2 + xFinal_8 => xFinal_1 + 3 xFinal_2 + xFinal_8; 1
       R_28: xFinal_1 + xFinal_2 + xFinal_8 => xFinal_1 + 2 xFinal_2 + xFinal_8; 1
       R_29: xFinal_2 => 2 xFinal_2; 1
       R_30: xFinal_2 => ; 1
       R_31: xFinal_2 => xFinal_1; 1
       R_32: xFinal_1 + xFinal_2 + xFinal_8 => 2 xFinal_1 + xFinal_2 + xFinal_8; 1                                                                          
       R_33: xFinal_1 + xFinal_2 + xFinal_8 => xFinal_2 + xFinal_8; 1
       R_34: xFinal_1 + xFinal_2 + xFinal_7 + xFinal_8 => xFinal_2 + xFinal_7 + xFinal_8; 1
       R_35: xFinal_1 => ; 1
                                                                                     

      
        // Species initializations:
        xFinal_1 = 362;
        xFinal_2 = 77;
        xFinal_3 = 61;
        xFinal_4 = 238;
        xFinal_5 = 119;
        xFinal_6 = 185;
        xFinal_7 = 6459;
        xFinal_8 = 32098;
        xFinal_9 = 20536;
        xFinal_10 = 79788;
        xFinal_11 = 0;
        xFinal_12 = 77633;
        """
        rr = te.loada(model)
        stoichiometry = Stoichiometry(None, roadrunner=rr)
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    unittest.main()