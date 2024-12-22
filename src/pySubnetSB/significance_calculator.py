'''Calculates the significance level of finding an induced subnetwork in a target network.'''

"""
The null distribution are randomly generated models tuned to the BioModels repository.
A monte carlo simulation is used to generate the null distribution, and test models to see if
a reference network is present in the target network.
"""

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.network import Network  # type: ignore

import collections
import numpy as np
import pandas as pd  # type: ignore
from typing import List  # type: ignore
import tqdm # type: ignore


NUM_REFERENCE_SPECIES = "num_reference_species"
NUM_REFERENCE_REACTION = "num_reference_reaction"
NUM_TARGET_SPECIES = "num_target_species"
NUM_TARGET_REACTION = "num_target_reaction"
FRAC_INDUCED = "frac_induced"
FRAC_TRUNCATED = "frac_truncated"


SignificanceCalculatorResult = collections.namedtuple("SignificanceCalculatorResult", 
  ["num_reference_species", "num_reference_reaction", "num_target_species", "num_target_reaction",
   "num_iteration", "max_num_assignment", "identity",
   "num_induced", "num_truncated", "frac_induced", "frac_truncated"])


class SignificanceCalculator(object):

    def __init__(self, reference_network:Network, num_target_reaction:int, num_target_species:int,
                 identity:str=cn.ID_WEAK)->None:
        self.reference_network = reference_network
        self.num_target_reaction = num_target_reaction
        self.num_target_species = num_target_species
        self.identity = identity

    def calculate(self, num_iteration:int,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->SignificanceCalculatorResult:
        """
        Calculates the significance level of finding an induced subnetwork in a target network.

        Args:
            num_iteration (int): Number of iterations
            max_num_assignment (int): Maximum number of assignment pairs

        Returns:
            SignificanceCalculatorResult
        """
        num_induced = 0
        num_truncated = 0
        for _ in tqdm.tqdm(range(num_iteration), desc="iteration"):
            target_network = Network.makeRandomNetworkByReactionType(self.num_target_reaction,
                  self.num_target_reaction)
            result = self.reference_network.isStructurallyIdentical(target_network,
                  identity=self.identity, is_subnet=True,
                    max_num_assignment=max_num_assignment, is_report=False)
            num_induced += 1 if result else 0
            num_truncated += 1 if result.is_truncated else 0
        # Calculate the significance level
        return SignificanceCalculatorResult(
            num_reference_species=self.reference_network.num_species,
            num_reference_reaction=self.reference_network.num_reaction,
            num_target_species=self.num_target_species,
            num_target_reaction=self.num_target_reaction,
            num_iteration=num_iteration,
            max_num_assignment=max_num_assignment,
            identity=self.identity,
            num_induced=num_induced,
            num_truncated=num_truncated,
            frac_induced=num_induced/num_iteration,
            frac_truncated=num_truncated/num_iteration)
    
    @classmethod
    def generateNullDistribution(cls, num_iteration:int=1000, identity:str=cn.ID_WEAK,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
          reference_sizes:List[int]=[2*n for n in range(1, 11)],
          target_sizes:List[int]=[5*n + 20 for n in range(18)])->pd.DataFrame:
        """
        Generates a null distribution of finding an induced subnetwork in a target network.

        Args:
            reference_network (Network): Reference network
            num_target_reaction (int): Number of target species
            num_target_reaction (int): Number of target reactions
            num_iteration (int): Number of iterations
            identity (str): Identity type
            max_num_assignment (int): Maximum number of assignment pairs

        Returns:
            pd.DataFrame: Columns
                reference_size, target_size, frac_induced, frac_truncated
        """
        REFERENCE_SIZE = "reference_size"
        TARGET_SIZE = "target_size"
        dct:dict = {n: [] for n in [REFERENCE_SIZE, TARGET_SIZE, FRAC_INDUCED, FRAC_TRUNCATED]}
        for reference_size in reference_sizes:
            for target_size in target_sizes:
                reference_network = Network.makeRandom(reference_size, reference_size)
                calculator = SignificanceCalculator(reference_network, target_size, target_size,
                      identity=identity)
                result = calculator.calculate(num_iteration, max_num_assignment)
                dct[REFERENCE_SIZE].append(reference_size)
                dct[TARGET_SIZE].append(target_size)
                dct[FRAC_INDUCED].append(result.frac_induced)
                dct[FRAC_TRUNCATED].append(result.frac_truncated)
        # Create the DataFrame
        df = pd.DataFrame(dct)
        return df