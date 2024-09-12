'''Benchmarks for isStructurallyIdentical and isStructurallyIdenticalSubnet.'''

import sirn.constants as cn # type: ignore
from sirn.network import Network # type: ignore
from sirn.assignment_pair import AssignmentPair  # type: ignore


import collections
import numpy as np
import time
from typing import List, Tuple, Optional


# An experiment defines the inputs to a benchark to run.
ExperimentResult = collections.namedtuple('ExperimentResult',
      'num_experiment runtimes num_success num_truncated')


##############################
class Experiment(object):

    def __init__(self, reference:Network, target:Network, assignment_pair:AssignmentPair):
        self.reference = reference
        self.target = target
        self.assignment_pair = assignment_pair

    def __repr__(self):
        #####
        def networkRepr(network:Network):
            return f"(num_species: {network.num_species}, num_reaction: {network.num_reaction})"
        #####
        return f"reference: {networkRepr(self.reference)}, target: {networkRepr(self.target)}"


##############################
class BenchmakrRunner(object):

    def __init__(self, reference_size:int=3, expansion_factor:int=1, identity=cn.ID_WEAK):
        """
        Args:
            reference_size (int)
            expansion_factor (int): Integer factor for size of target relative to reference
            identity (str)
        """
        self.reference_size = reference_size
        self.expansion_factor = expansion_factor
        self.identity = identity

    def makeExperiment(self):
        """Construct the elements of an experiment.

        Returns:
            Experiment
        """
        #####
        def lengthenArray(array:np.ndarray)->np.ndarray:
            # Orignal array is in the upper left
            longer_arr = np.concatenate([array]*target_size, axis=0)
            return np.hstack([array, longer_arr])
        #####
        def permute(arr:np.ndarray, assignment_pair)->np.ndarray:
            arr = arr.copy()
            arr = arr[assignment_pair.species_idxs, :]
            return arr[:, assignment_pair.reaction_idxs]
        #####
        reference = Network.makeRandomNetworkByReactionType(self.reference_size, is_prune_species=False)
        target_size = self.reference_size*self.expansion_factor
        filler_size = target_size - self.reference_size
        if filler_size == 0:
            # Do not expand the target
            target, assignment_pair = reference.permute()
        else:
            # Expand the target
            #   Create the left part of the expanded target
            left_filler_arr = np.zeros((filler_size, self.reference_size))
            xreactant_arr = np.vstack([reference.reactant_mat.values, left_filler_arr])
            xproduct_arr = np.vstack([reference.product_mat.values, left_filler_arr])
            #  Create the expanded arrays
            right_filler = Network.makeRandomNetworkByReactionType(num_reaction=filler_size,
                  num_species=target_size, is_prune_species=False)
            xreactant_arr = np.hstack([xreactant_arr, right_filler.reactant_mat.values])
            xproduct_arr = np.hstack([xproduct_arr, right_filler.product_mat.values])
            xtarget = Network(xreactant_arr, xproduct_arr)
            #  Merge the left and right parts
            target, assignment_pair = xtarget.permute()
        return Experiment(reference=reference, target=target, assignment_pair=assignment_pair)