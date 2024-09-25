'''Evaluates the effectiveness of reaction and species constraints.'''

"""
Key data structures:
    BenchmarkResult is a dataframe
        Index: index of the network used
        Columns:
            time - execution time
            num_permutations - number of permutations

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
    def __init__(self, num_species:int, num_reaction:int, num_iteration:int, is_subset:bool=False):
        self.num_reaction = num_reaction
        self.num_species = num_species
        self.num_iteration = num_iteration
        self.is_subset = is_subset
        # Calculated
        self.networks = [Network.makeRandomNetworkByReactionType(self.num_reaction, self.num_species)
                for _ in range(num_iteration)]
        self.benchmark_result_df = NULL_DF  # Updated with result of run

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
        for network in self.networks:
            # Species
            start = time.time()
            constraint = constraint_cls(network.reactant_nmat, network.product_nmat,
                    is_subset=self.is_subset)
            compatibility_collection = constraint.makeCompatibilityCollection(constraint)
            times.append(time.time() - start)
            num_permutations.append(compatibility_collection.num_permutation)
        self.benchmark_result_df = pd.DataFrame({C_TIME: times, C_NUM_PERMUTATION: num_permutations})
        return self.benchmark_result_df

    @classmethod 
    def plotConstraintStudy(cls, num_species:int, num_reaction:int):
        """Plot the results of a study of constraints.

        Args:
            num_reaction (int): Number of reactions
            num_species (int): Number of species
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
            title = f"{node_str}; is_subset={is_subset}"
            ax.set_title(title)
            ax.set_ylim(0, 10)
            ax.set_xlim(0, 1)
            if pos in [0, 2]:
                ax.set_ylabel('Log10 permutations')
            else:
                ax.set_yticks([])
            if pos in [2, 3]:
                ax.set_xlabel('Culmulative networks')
            else:
                ax.set_xticks([])
    #####
        num_iteration = 1000
        fig, axes = plt.subplots(2, 2)
        fig.suptitle(f"CDF of permutations for {num_species} species and {num_reaction} reactions")
        # Collect data and construct plots
        pos = 0
        for is_species in [False, True]:
            if is_species:
                node_str = 'Species'
            else:
                node_str = 'Reaction'
            for is_subset in [False, True]:
                benchmark = cls(num_species, num_reaction, num_iteration, is_subset=is_subset)
                benchmark.run(is_species=is_species)
                doPlot(benchmark.benchmark_result_df, node_str, is_subset, pos=pos)
                pos += 1
        plt.show()
    

if __name__ == '__main__':
    Benchmark.plotConstraintStudy(20, 20)