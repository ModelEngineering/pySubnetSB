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

from sirn.reaction_constraint import ReactionConstraint  # type: ignore
from sirn.species_constraint import SpeciesConstraint    # type: ignore
from sirn.network import Network                         # type: ignore

import collections
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import time
from typing import List

NULL_DF = pd.DataFrame()
C_TIME = 'time'
C_LOG10_NUM_PERMUTATION = 'num_permutation'


# A study result is a container of the results of multiple benchmarks
StudyResult = collections.namedtuple('StudyResult', ['is_categorical_id', 'study_ids', 'benchmark_results'])
#  is_categorical_id: bool # True if study_ids are categorical
#  study_ids: List[str | float]
#  benchmark_results: List[pd.DataFrame]


class ConstraintBenchmark(object):
    def __init__(self, reference_size:int, fill_size:int=0, num_iteration:int=1000,
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
        self.fill_size = fill_size
        self.is_subset = is_subset
        # Calculated
        self.reference_networks = [Network.makeRandomNetworkByReactionType(self.num_reaction, self.num_species)
                for _ in range(num_iteration)]
        if is_subset and fill_size > 0:
            self.target_networks = [Network.fill(n, num_fill_reaction=self.num_reaction,
              num_fill_species=self.num_species) for n in self.reference_networks]
        self.target_networks = [Network.fill(n, num_fill_reaction=self.fill_size,
              num_fill_species=self.fill_size) for n in self.reference_networks]
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
        for reference_network, target_network in zip(self.reference_networks, self.target_networks):
            # Species
            start = time.time()
            reference_constraint = constraint_cls(
                    reference_network.reactant_nmat, reference_network.product_nmat,
                    is_subset=self.is_subset)
            if self.is_subset:
                target_constraint = constraint_cls(
                    target_network.reactant_nmat, target_network.product_nmat, is_subset=self.is_subset)
            else:
                target_constraint = reference_constraint
            compatibility_collection = reference_constraint.makeCompatibilityCollection(target_constraint)
            times.append(time.time() - start)
            num_permutations.append(compatibility_collection.log10_num_permutation)
        self.benchmark_result_df = pd.DataFrame({C_TIME: times, C_LOG10_NUM_PERMUTATION: num_permutations})
        return self.benchmark_result_df

    @classmethod 
    def plotConstraintStudy(cls, reference_size:int, fill_size:int, num_iteration:int=1000,
                            is_plot:bool=True, **kwargs):
        """Plot the results of a study of constraints.

        Args:
            reference_size (int): Size of the reference network
            fill_size (int): Size of the filler network
            num_iteration (int, optional): Number of iterations. Defaults to 1000.
            is_plot (bool, optional): Plot the results. Defaults to True.
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
        suptitle = f"CDF of permutations: reference_size={reference_size}; fill_size={fill_size}"
        fig.suptitle(suptitle)
        # Collect data and construct plots
        pos = 0
        benchmark_dct = {}
        for is_subset in [False, True]:
            for is_species in [False, True]:
                if is_species:
                    node_str = 'Spc.'
                else:
                    node_str = 'Rct.'
                benchmark_dct[is_species] = cls(reference_size, fill_size, num_iteration,
                      is_subset=is_subset, **kwargs)
                benchmark_dct[is_species].run(is_species=is_species)
                doPlot(benchmark_dct[is_species].benchmark_result_df, node_str, is_subset, pos=pos)
                pos += 1
            # Construct totals plot
            df = benchmark_dct[True].benchmark_result_df.copy()
            df += benchmark_dct[False].benchmark_result_df
            doPlot(df, "Total", is_subset, pos=pos)
            pos += 1
        if is_plot:
            plt.show()
    

if __name__ == '__main__':
    size = 20
    ConstraintBenchmark.plotConstraintStudy(size, fill_size=size, num_iteration=100, is_plot=True)