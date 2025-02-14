'''Calculates the significance level of finding an induced subnetwork in a target network.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.network import Network  # type: ignore

import collections
import numpy as np
import tqdm # type: ignore
from typing import List, Tuple, Optional


NUM_REFERENCE_SPECIES = "num_reference_species"
NUM_REFERENCE_REACTION = "num_reference_reaction"
NUM_TARGET_SPECIES = "num_target_species"
NUM_TARGET_REACTION = "num_target_reaction"
FRAC_INDUCED = "frac_induced"
FRAC_TRUNCATED = "frac_truncated"
# Default values
NUM_ITERATION = 1000


SignificanceCalculatorCoreResult = collections.namedtuple("SignificanceCalculatorCoreResult", 
  ["num_reference_species", "num_reference_reaction", "num_target_species", "num_target_reaction",
   "num_target_network", "max_num_assignment", "identity",
   "num_induced", "num_truncated", "frac_induced", "frac_truncated"])


class SignificanceCalculatorCore(object):

    def __init__(self, num_target_species:int, num_target_reaction:int, num_target_network:int):
        self.num_target_reaction = num_target_reaction
        self.num_target_species = num_target_species
        self.num_target_network = num_target_network
        target_networks = [Network.makeRandomNetworkByReactionType(num_target_reaction,
                num_target_species, is_exact=False) for _ in range(num_target_network)]
        self.target_networks = [n for n in target_networks
                if (n.num_species == num_target_species) and (n.num_reaction == num_target_reaction)]
        self.count_target_network = len(self.target_networks)
        self.target_dct:Optional[dict] = None

    def calculateSubnet(self, reference_network:Network, identity:str=cn.ID_WEAK, is_subnet:bool=True,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT, is_report:bool=True
          )->SignificanceCalculatorCoreResult:
        """
        Calculates the significance level of finding an induced subnetwork in a target network.

        Args:
            reference_network (Network): Reference network
            is_subnet (bool): If True, the target network is a subnet of the reference network
            identity (str): Identity type
            max_num_assignment (int): Maximum number of assignment pairs
            is_report (bool): If True, report progress

        Returns:
            SignificanceCalculatorCoreResult
        """
        num_induced = 0
        num_truncated = 0
        for target_network in tqdm.tqdm(self.target_networks, desc="iteration", disable=not is_report):
            result = reference_network.isStructurallyIdentical(target_network,
                  identity=identity, is_subnet=is_subnet,
                    max_num_assignment=max_num_assignment, is_report=False)
            num_induced += 1 if result else 0
            num_truncated += 1 if result.is_truncated else 0
        # Calculate the significance level
        return SignificanceCalculatorCoreResult(
            num_reference_species=reference_network.num_species,
            num_reference_reaction=reference_network.num_reaction,
            num_target_species=self.num_target_species,
            num_target_reaction=self.num_target_reaction,
            num_target_network=self.count_target_network,
            max_num_assignment=max_num_assignment,
            identity=identity,
            num_induced=num_induced,
            num_truncated=num_truncated,
            frac_induced=num_induced/self.count_target_network if self.count_target_network > 0 else np.nan,
            frac_truncated=num_truncated/self.count_target_network if self.count_target_network > 0 else np.nan)

    def calculateEqual(self, reference_network:Network, identity:str=cn.ID_WEAK,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT, is_report:bool=True
          )->SignificanceCalculatorCoreResult:
        """
        Calculates the significance level of finding a structurally identical target network.

        Args:
            reference_network (Network): Reference network
            identity (str): Identity type
            max_num_assignment (int): Maximum number of assignment pairs
            is_report (bool): If True, report progress

        Returns:
            SignificanceCalculatorCoreResult
        """
        if self.target_dct is None:
            self.target_dct = {}
            for target_network in self.target_networks:
                key = target_network.network_hash
                if not key in self.target_dct:
                    self.target_dct[key] = []
                self.target_dct[key].append(target_network)
        # Look for structurally identical networks
        num_induced = 0
        num_truncated = 0
        key = reference_network.network_hash
        if key in self.target_dct:
            for target_network in tqdm.tqdm(self.target_dct[key], desc="iteration", disable=not is_report):
                result = reference_network.isStructurallyIdentical(target_network, identity=identity,
                    is_subnet=False,
                    max_num_assignment=max_num_assignment, is_report=False)
                num_induced += 1 if result else 0
                num_truncated += 1 if result.is_truncated else 0
        # Return the result
        if self.count_target_network > 0:
            frac_induced = num_induced/self.count_target_network
            frac_truncated = num_truncated/self.count_target_network
        else:
            frac_induced = np.nan
            frac_truncated = np.nan
        return SignificanceCalculatorCoreResult(
            num_reference_species=reference_network.num_species,
            num_reference_reaction=reference_network.num_reaction,
            num_target_species=self.num_target_species,
            num_target_reaction=self.num_target_reaction,
            num_target_network=self.count_target_network,
            max_num_assignment=max_num_assignment,
            identity=identity,
            num_induced=num_induced,
            num_truncated=num_truncated,
            frac_induced=frac_induced,
            frac_truncated=frac_truncated,
        )