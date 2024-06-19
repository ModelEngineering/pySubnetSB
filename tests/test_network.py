import sirn.constants as cn  # type: ignore
from sirn.pmatrix import PMatrix # type: ignore
from sirn.network import Network # type: ignore

import copy
import os
import pandas as pd  # type: ignore
import numpy as np # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
NETWORK_NAME = "test"
MODEL = """
J0: S1 -> S2; k1*S1;
J1: S2 -> S3; k2*S2;

k1 = 1
k2 = 1
S1 = 10
S2 = 0
S3 = 0
"""
NETWORK = Network.makeAntimony(MODEL, network_name=NETWORK_NAME, is_simple_stoichiometry=False)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = copy.deepcopy(NETWORK)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.network.network_name, NETWORK_NAME)
        for pmatrix in [self.network.reactant_pmatrix, self.network.product_pmatrix]:
            self.assertTrue(isinstance(pmatrix, PMatrix))



if __name__ == '__main__':
    unittest.main()