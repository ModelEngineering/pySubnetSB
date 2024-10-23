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
"""

from sirn.reaction_constraint import ReactionConstraint  # type: ignore
from sirn.species_constraint import SpeciesConstraint    # type: ignore
from sirn.network import Network                         # type: ignore

import collections
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import time
from typing import List

NULL_DF = pd.DataFrame()
C_TIME = 'time'
C_LOG10_NUM_PERMUTATION = 'log10num_permutation'
C_NUM_REFERENCE = 'num_reference'
C_NUM_TARGET = 'num_target'


# A study result is a container of the results of multiple benchmarks
StudyResult = collections.namedtuple('StudyResult', ['is_categorical_id', 'study_ids', 'benchmark_results'])
#  is_categorical_id: bool # True if study_ids are categorical
#  study_ids: List[str | float]
#  benchmark_results: List[pd.DataFrame]


class ConstraintBenchmark(object):
    def __init__(self, reference_size:int, fill_size:int=0, num_iteration:int=1000):
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
        # Calculated
        self.reference_networks = [Network.makeRandomNetworkByReactionType(self.num_reaction, self.num_species)
                for _ in range(num_iteration)]
        if fill_size > 0:
            self.target_networks = [n.fill(num_fill_reaction=self.num_reaction,
              num_fill_species=self.num_species) for n in self.reference_networks]
        self.target_networks = [n.fill(num_fill_reaction=self.fill_size,
              num_fill_species=self.fill_size) for n in self.reference_networks]
        self.benchmark_result_df = NULL_DF  # Updated with result of run

    @staticmethod 
    def _getConstraintClass(is_species:bool=True):
        if is_species:
            return SpeciesConstraint
        else:
            return ReactionConstraint

    def run(self, is_subset:bool=True, is_species:bool=True)->pd.DataFrame:
        """Evaluates the effectiveness of reaction and species constraints.

        Args:
            is_subset (bool, optional): is subset constraint
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
                    is_subset=is_subset)
            if is_subset:
                target_constraint = constraint_cls(
                    target_network.reactant_nmat, target_network.product_nmat, is_subset=is_subset)
            else:
                new_target_network, _ = reference_network.permute()
                target_constraint = constraint_cls(
                    new_target_network.reactant_nmat, new_target_network.product_nmat,
                    is_subset=is_subset)
            compatibility_collection = reference_constraint.makeCompatibilityCollection(target_constraint)
            times.append(time.time() - start)
            num_permutations.append(compatibility_collection.log10_num_assignment)
        self.benchmark_result_df = pd.DataFrame({C_TIME: times, C_LOG10_NUM_PERMUTATION: num_permutations})
        return self.benchmark_result_df

    @classmethod 
    def plotConstraintStudy(cls, reference_size:int, fill_size:int, num_iteration:int,
          is_plot:bool=True, **kwargs):
        """Plot the results of a study of constraints.

        Args:
            reference_size (int): size of the reference network (species, reaction)
            fill_size (int): size of the filler network (species, reaction) used in subsets
            num_iteration (int): number of iterations
            is_plot (bool, optional): Plot the results. Defaults to True.
            kwargs: constructor options
        """
        benchmark = ConstraintBenchmark(reference_size, fill_size=fill_size, num_iteration=num_iteration)
        def doPlot(df:pd.DataFrame, node_str:str, is_subset:bool=False, pos:int=0):
            ax = axes.flatten()[pos]
            xv = np.array(range(len(df)))
            yv = df[C_LOG10_NUM_PERMUTATION].values.copy()
            sel = yv == 0
            yv[sel] = 1
            yv = yv
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
        fig, axes = plt.subplots(2, 3)
        suptitle = f"CDF of permutations: reference_size={benchmark.num_reaction}; fill_size={benchmark.fill_size}"
        fig.suptitle(suptitle)
        # Collect data and construct plots
        pos = 0
        for is_subset in [False, True]:
            dataframe_dct:dict = {}
            for is_species in [False, True]:
                if is_species:
                    node_str = 'Spc.'
                else:
                    node_str = 'Rct.'
                dataframe_dct[is_species] = benchmark.run(is_species=is_species, is_subset=is_subset)
                doPlot(dataframe_dct[is_species], node_str, is_subset, pos=pos)
                pos += 1
            # Construct totals plot
            df = dataframe_dct[True].copy()
            df += dataframe_dct[False]
            doPlot(df, "Total", is_subset, pos=pos)
            pos += 1
        if is_plot:
            plt.show()

    def plotHeatmap(self, num_references:List[int], num_targets:List[int], percentile:int=50,
                    num_iteration:int=20, is_plot:bool=True)->pd.DataFrame:
        """Plot a heatmap of the log10 of number of permutations.

        Args:
            num_references (List[int]): number of reference networks
            num_targets (List[int]): number of target networks
            percentile (int): percentile of distribution of the log number of permutation

        Returns:
            pd.DataFrame: _description_
        """
        # Construct the dataj
        data_dct:dict = {C_NUM_REFERENCE: [], C_NUM_TARGET: [], C_LOG10_NUM_PERMUTATION: []}
        for reference_size in num_references:
            for target_size in num_targets:
                fill_size = target_size - reference_size
                data_dct[C_NUM_REFERENCE].append(reference_size)
                data_dct[C_NUM_TARGET].append(target_size)
                is_subset = fill_size > 0
                if fill_size < 0:
                    result = np.nan
                else:
                    fill_size = max(1, fill_size)
                    benchmark = ConstraintBenchmark(reference_size, fill_size=fill_size, num_iteration=num_iteration)
                    df_species = benchmark.run(is_species=True, is_subset=is_subset)
                    df_reaction = benchmark.run(is_species=False, is_subset=is_subset)
                    df = df_species + df_reaction
                    result = np.percentile(df[C_LOG10_NUM_PERMUTATION].values, percentile)
                data_dct[C_LOG10_NUM_PERMUTATION].append(result)
        # Construct the dataframe
        df = pd.DataFrame(data_dct)
        df = df.rename(columns={C_NUM_REFERENCE: 'Reference', C_NUM_TARGET: 'Target'})
        df[C_LOG10_NUM_PERMUTATION] = np.round(df[C_LOG10_NUM_PERMUTATION].astype(float), 1)
        pivot_df = df.pivot(columns='Reference', index='Target', values=C_LOG10_NUM_PERMUTATION)
        pivot_df = pivot_df.sort_index(ascending=False)
        # Plot
        ax = sns.heatmap(pivot_df, annot=True, fmt="g", cmap='Reds', vmin=0, vmax=10,
                          cbar_kws={'label': 'log10 number of permutations'})
        ax.set_title(f'percentile: {percentile}')
        if is_plot:
            plt.show()
        #
        return df
    

if __name__ == '__main__':
    size = 6
    #ConstraintBenchmark.plotConstraintStudy(size, 9*size, num_iteration=50, is_plot=True)
    benchmark = ConstraintBenchmark(6, 6, 20)
    _ = benchmark.plotHeatmap(list(range(4, 22, 2)), list(range(10, 105, 5)), percentile=50,
                                        num_iteration=300)