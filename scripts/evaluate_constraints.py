'''Evaluates the effectiveness of reaction and species constraints.'''

"""
Evaluates subset detection by seeing if the reference network can be found when it is combined
with a filer network.

Key data structures:
    BenchmarkResult is a dataframe
        Index: index of the network used
        Columns:
            time - execution time
            num_permutations - number of permutations

To do
1. Tests

"""

from src.sirn.reaction_constraint import ReactionConstraint
from src.sirn.species_constraint import SpeciesConstraint
from src.sirn.network import Network

import collections
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import time
from typing import List

NULL_DF = pd.DataFrame()
C_TIME = 'time'
C_NUM_PERMUTATION = 'num_permutation'


# A study result is a container of the results of multiple benchmarks
StudyResult = collections.namedtuple('StudyResult', ['is_categorical_id', 'study_ids', 'benchmark_results'])
#  is_categorical_id: bool # True if study_ids are categorical
#  study_ids: List[str | float]
#  benchmark_results: List[pd.DataFrame]


class Benchmark(object):
    def __init__(self, reference_size:int, fillter_size:int, num_iteration:int,
                 is_subset:bool=False):
        """
        Args:
            reference_size (int): size of the reference network (species, reaction)
            filler_size (int): size of the filler network (species, reaction) used in subsets
            is_subset (bool, optional): _description_. Defaults to False.
        """
        self.num_reaction = reference_size
        self.num_species = reference_size
        self.num_iteration = num_iteration
        self.filler_size = fillter_size
        self.is_subset = is_subset
        # Calculated
        self.reference_networks = [Network.makeRandomNetworkByReactionType(self.num_reaction, self.num_species)
                for _ in range(num_iteration)]
        self.filler_networks = [Network.makeRandomNetworkByReactionType(
                self.filler_size, self.filler_size)
                for _ in range(num_iteration)]
        self.benchmark_result_df = NULL_DF  # Updated with result of run

    @staticmethod
    def makeFillerNetwork(reference_network:Network, filler_network:Network)->Network:
        # Creates a supernetwork with the reference in the upper left corner of the matrices
        # and the filler network in the bottom right. Then, randomize.
        num_reference_species = reference_network.num_species
        num_filler_species = filler_network.num_species
        num_reference_reaction = reference_network.num_reaction
        num_filler_reaction = filler_network.num_reaction
        right_hpad_arr = np.zeros((num_reference_species, num_filler_reaction))
        right_hpad_arr = np.zeros((num_reference_species, num_reference_reaction))
        left_hpad_arr = np.zeros((num_filler_species, num_reference_reaction))
        #####
        def makeTargetArray(reference_arr:np.ndarray, filler_arr:np.ndarray)->np.ndarray:
            target_reactant_arr = np.hstack([reference_arr, right_hpad_arr])
            target_reactant_arr = np.vstack([target_reactant_arr, left_hpad_arr])
            target_reactant_arr = np.vstack([target_reactant_arr, left_hpad_arr])
            return np.hstack([left_hpad_arr, filler_arr])
        ##### 
        # Construct the reactant array so that reference is upper left and filler is lower right
        target_reactant_arr = makeTargetArray(reference_network.reactant_nmat.values,
              filler_network.reactant_nmat.values)
        target_product_arr = makeTargetArray(reference_network.product_nmat.values,
              filler_network.product_nmat.values)
        target_network = Network(target_reactant_arr, target_product_arr)
        return target_network.permute()

    @staticmethod 
    def _getConstraintClass(is_species:bool=True):
        if is_species:
            return SpeciesConstraint
        else:
            return ReactionConstraint

    def run(self, is_species:bool=True)->pd.DataFrame:
        """Evaluates the effectiveness of reaction and species constraints.

        Args:
            is_species (bool, optional): is species constraint

        Returns:
            pd.DataFrame: _description_
        """
        
        times:list = []
        num_permutations:list = []
        constraint_cls = self._getConstraintClass(is_species)
        for reference_network, filler_network in zip(self.reference_networks, self.filler_networks):
            # Species
            start = time.time()
            reference_constraint = constraint_cls(
                    reference_network.reactant_nmat, reference_network.product_nmat,
                    is_subset=self.is_subset)
            if self.is_subset:
                target_network = self._makeTargetArray(reference_network, filler_network)
                target_constraint = constraint_cls(target_network.reactant_nmat, target_network.product_nmat)
            else:
                target_constraint = reference_constraint
            compatibility_collection = reference_constraint.makeCompatibilityCollection(target_constraint)
            times.append(time.time() - start)
            num_permutations.append(compatibility_collection.num_permutation)
        self.benchmark_result_df = pd.DataFrame({C_TIME: times, C_NUM_PERMUTATION: num_permutations})
        return self.benchmark_result_df

    @classmethod 
    def plotConstraintStudy(cls, num_species:int, num_reaction:int, **kwargs):
        """Plot the results of a study of constraints.

        Args:
            num_reaction (int): Number of reactions
            num_species (int): Number of species
            kwargs: constructor options
        """
        def doPlot(df:pd.DataFrame, node_str:str, is_subset:bool=False, pos:int=0):
            ax = axes.flatten()[pos]
            xv = np.array(range(len(df)))
            yv = df['num_permutation'].values.copy()
            sel = yv == 0
            yv[sel] = 1
            yv = np.log10(yv)
            yv = np.sort(yv)
            xv = xv/len(yv)
            ax.plot(xv, yv)
            title = 'Subset' if is_subset else 'Full'
            title = node_str + ' ' + title
            ax.set_title(title)
            ax.set_ylim(0, 10)
            ax.set_xlim(0, 1)
            if pos in [0, 3]:
                ax.set_ylabel('Log10 permutations')
            else:
                ax.set_yticks([])
            if pos in [3, 4, 5]:
                ax.set_xlabel('Culmulative networks')
            else:
                ax.set_xticks([])
            ax.plot([0, 1], [8, 8], 'k--')
    #####
        num_iteration = 1000
        fig, axes = plt.subplots(2, 3)
        expansion_factor = kwargs.get('expansion_factor', 1)
        suptitle = f"CDF of permutations for; #species={num_species}; "
        suptitle += f"#reaction={num_reaction}; expf={expansion_factor}"
        fig.suptitle(suptitle)
        # Collect data and construct plots
        pos = 0
        benchmark_dct = {}
        for is_subset in [False, True]:
            if is_subset:
                expf = expansion_factor
            else:
                expf = 1
            for is_species in [False, True]:
                if is_species:
                    node_str = 'Spc.'
                else:
                    node_str = 'Rct.'
                benchmark_dct[is_species] = cls(num_species, num_reaction, num_iteration,
                      is_subset=is_subset, **kwargs)
                benchmark_dct[is_species].run(is_species=is_species)
                doPlot(benchmark_dct[is_species].benchmark_result_df, node_str, is_subset, pos=pos)
                pos += 1
            # Construct totals plot
            df = benchmark_dct[True].benchmark_result_df.copy()
            df += benchmark_dct[False].benchmark_result_df
            doPlot(df, "Total", is_subset, pos=pos)
            pos += 1
        plt.show()
    

if __name__ == '__main__':
    Benchmark.plotConstraintStudy(5, 5, expansion_factor=3)