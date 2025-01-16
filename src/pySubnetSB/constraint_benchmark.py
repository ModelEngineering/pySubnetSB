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
   1. Construct bar plots where x-axis indicates the constraints, and y-axis is log10 of num_assignments
   2. Do (1) for 3 network sizes and species, reactions. 6 plots total.  
   3. Should I consider all combinations of options?
"""

import scipy.special  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore
from pySubnetSB.constraint_option_collection import SpeciesConstraintOptionCollection  # type: ignore
from pySubnetSB.constraint_option_collection import ReactionConstraintOptionCollection  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint    # type: ignore
from pySubnetSB.network import Network      # type: ignore
import pySubnetSB.constants as cn # type: ignore

import collections
import matplotlib.pyplot as plt # type: ignore
import math
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import scipy
import seaborn as sns  # type: ignore
import time
from typing import List, Optional

NULL_DF = pd.DataFrame()
C_TIME = 'time'
C_LOG10_NUM_PERMUTATION = 'log10num_permutation'
C_NUM_REFERENCE = 'num_reference'
C_NUM_TARGET = 'num_target'

DimensionResult = collections.namedtuple('DimensionResult', ['dataframe', 'short_to_long_dct'])
EvaluateConstraintsResult = collections.namedtuple('EvaluateConstraintsResult',
      ['reference_size', 'target_size', 'species_dimension_result', 'reaction_dimension_result'])


# A study result is a container of the results of multiple benchmarks
StudyResult = collections.namedtuple('StudyResult', ['is_categorical_id', 'study_ids', 'benchmark_results'])
#  is_categorical_id: bool # True if study_ids are categorical
#  study_ids: List[str | float]
#  benchmark_results: List[pd.DataFrame]


class ConstraintBenchmark(object):
    def __init__(self, reference_size:int, fill_size:int=0, num_iteration:int=1000,
          is_contains_reference:bool=True)->None:
        """
        Args:
            reference_size (int): size of the reference network (species, reaction)
            fill_size (int): size of the filler network (species, reaction) used in subsets
            is_contains_reference (bool, Optional): target network contains reference network. Defaults to True.
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
        if is_contains_reference:
            self.target_networks = [n.fill(num_fill_reaction=self.fill_size,
                num_fill_species=self.fill_size) for n in self.reference_networks]
        else:
            target_size = reference_size + self.fill_size
            self.reference_networks = [Network.makeRandomNetworkByReactionType(target_size, target_size)
                for _ in range(num_iteration)]
        self.benchmark_result_df = NULL_DF  # Updated with result of run

    @staticmethod 
    def _getConstraintClass(is_species:bool=True):
        if is_species:
            return SpeciesConstraint
        else:
            return ReactionConstraint

    def run(self, is_subnet:bool=True, is_species:bool=True)->pd.DataFrame:
        """Evaluates the effectiveness of reaction and species constraints.

        Args:
            is_subnet (bool, optional): is subset constraint
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
                    is_subnet=is_subnet)
            if is_subnet:
                target_constraint = constraint_cls(
                    target_network.reactant_nmat, target_network.product_nmat, is_subnet=is_subnet)
            else:
                new_target_network, _ = reference_network.permute()
                target_constraint = constraint_cls(
                    new_target_network.reactant_nmat, new_target_network.product_nmat,
                    is_subnet=is_subnet)
            compatibility_collection = reference_constraint.makeCompatibilityCollection(
                  target_constraint).compatibility_collection
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
        def doPlot(df:pd.DataFrame, node_str:str, is_subnet:bool=False, pos:int=0):
            ax = axes.flatten()[pos]
            xv = np.array(range(len(df)))
            yv = df[C_LOG10_NUM_PERMUTATION].values.copy()
            sel = yv == 0
            yv[sel] = 1
            yv = yv
            yv = np.sort(yv)
            xv = xv/len(yv)
            ax.plot(xv, yv)
            title = 'Subset' if is_subnet else 'Full'
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
        for is_subnet in [False, True]:
            dataframe_dct:dict = {}
            for is_species in [False, True]:
                if is_species:
                    node_str = 'Spc.'
                else:
                    node_str = 'Rct.'
                dataframe_dct[is_species] = benchmark.run(is_species=is_species, is_subnet=is_subnet)
                doPlot(dataframe_dct[is_species], node_str, is_subnet, pos=pos)
                pos += 1
            # Construct totals plot
            df = dataframe_dct[True].copy()
            df += dataframe_dct[False]
            doPlot(df, "Total", is_subnet, pos=pos)
            pos += 1
        if is_plot:
            plt.show()

    @classmethod
    def plotHeatmap(cls, num_references:List[int], num_targets:List[int], percentile:int=50,
          num_iteration:int=20, is_contains_reference=True, is_plot:bool=True,
          is_no_constraint:bool=False, title:Optional[str]=None)->plt.Axes:
        """Plot a heatmap of the log10 of number of permutations.

        Args:
            num_references (List[int]): number of reference networks
            num_targets (List[int]): number of target networks
            percentile (int): percentile of distribution of the log number of permutation
            is_plot (bool, optional): plot the heatmap. Defaults to True.
            is_not_constraint (bool, optional): True if no constraints are applied. Defaults to False.  
            title (str, optional): title of the plot. Defaults to None.

        Returns:
            matplotlib.axes._axes.Axes: _description_
        """
        #####
        def calculate(is_species:bool)->pd.DataFrame:
            df = benchmark.run(is_species=is_species, is_subnet=is_subnet)
            df[C_LOG10_NUM_PERMUTATION] = [max(v, 0) for v in df[C_LOG10_NUM_PERMUTATION]]
            return df
        #####
        # Construct the dataj
        C_LOG10_P10 = 'log10_num_permutation_p10'
        C_LOG10_P90 = 'log10_num_permutation_p90'
        C_LABEL = 'label'
        data_dct:dict = {C_NUM_REFERENCE: [], C_NUM_TARGET: [], C_LOG10_NUM_PERMUTATION: [],
              C_LOG10_P10: [], C_LOG10_P90: [], C_LABEL: []}
        for reference_size in num_references:
            for target_size in num_targets:
                fill_size = target_size - reference_size
                data_dct[C_NUM_REFERENCE].append(reference_size)
                data_dct[C_NUM_TARGET].append(target_size)
                is_subnet = fill_size > 0
                if fill_size < 0:
                    result = np.nan
                    percentile_10 = np.nan
                    percentile_90 = np.nan
                else:
                    fill_size = max(1, fill_size)
                    if is_no_constraint:
                        result = 2*cls.calculateNoconstraintLog10NumAssignment(reference_size, target_size)
                        percentile_10 = result
                        percentile_90 = result
                    else:
                        benchmark = cls(reference_size, fill_size=fill_size,
                            is_contains_reference=is_contains_reference, num_iteration=num_iteration)
                        df_species = calculate(is_species=True)
                        df_reaction = calculate(is_species=False)
                        df = df_species + df_reaction
                        result = np.percentile(df[C_LOG10_NUM_PERMUTATION].values, percentile)
                        # Calculate multiple percentiles
                        percentile_10 = np.percentile(df[C_LOG10_NUM_PERMUTATION].values, 10)
                        percentile_90 = np.percentile(df[C_LOG10_NUM_PERMUTATION].values, 90)
                data_dct[C_LOG10_NUM_PERMUTATION].append(np.round(result, 1))
                if np.isnan(percentile_10):
                    data_dct[C_LABEL].append("")
                else:
                    if is_no_constraint:
                        data_dct[C_LABEL].append(f"{np.round(result, 1)}")
                    else:
                        data_dct[C_LABEL].append(
                            f"{np.round(result, 1)}\n({np.round(percentile_10, 1)}, {np.round(percentile_90, 1)})")
                data_dct[C_LOG10_P10].append(np.round(percentile_10, 1))
                data_dct[C_LOG10_P90].append(np.round(percentile_90, 1))
        # Construct the dataframe
        df = pd.DataFrame(data_dct)
        df = df.rename(columns={C_NUM_REFERENCE: 'Reference', C_NUM_TARGET: 'Target'})
        if percentile > 0:
            df[C_LOG10_NUM_PERMUTATION] = np.round(df[C_LOG10_NUM_PERMUTATION].astype(float), 1)
        else:
            df[C_LOG10_NUM_PERMUTATION] = [np.round(v, 1) for v in df[C_LOG10_NUM_PERMUTATION].astype(float)]
        pivot_df = df.pivot(columns='Reference', index='Target', values=C_LOG10_NUM_PERMUTATION)
        pivot_df = pivot_df.sort_index(ascending=False)
        label_df = df.pivot(columns='Reference', index='Target', values=C_LABEL)
        label_df = label_df.sort_index(ascending=False)
        # Plot
        ax = sns.heatmap(pivot_df, annot=label_df.values, fmt="", cmap='Reds', vmin=0, vmax=15,
                          cbar_kws={'label': 'log10 number of permutations'})
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f'percentile: {percentile}th (10th, 90th)')
        if is_plot:
            plt.show()
        #
        return ax
    
    @staticmethod
    def calculateNoconstraintLog10NumAssignment(reference_size:int, target_size:int)->float:
        """
        Calculate the log10 of the number of assignments when no constraints are applied.

        Args:
            reference_size (int): _description_
            target_size (int): _description_

        Returns:
            float: _description_
        """
        count = math.comb(target_size, reference_size)*scipy.special.factorial(reference_size)
        return np.log10(count)
    
    def compareConstraints(self, is_subnet:bool=True)->EvaluateConstraintsResult:
        """
        Selects random reference networks and random targets with the reference network embedded.

        Args:
            is_subnet (bool, optional): reference is a subnet of target if True. Defaults to True.

        Returns:
            EvaluateConstraintsResult
                is_species: True if species constraint
                reference_size: size of the reference network
                target_size: size of the target network
                reaction_df: dataframe of the results
                species_df: dataframe of the results
                short_to_long_dct: mapping of short names to long names
        """
        # Initialize result dictionaries
        species_result_dct:dict = {o.collection_short_name: []
              for o in SpeciesConstraintOptionCollection().iterator()}
        reaction_result_dct:dict = {o.collection_short_name: []
              for o in ReactionConstraintOptionCollection().iterator()}
        # Calculate the number of assignments if no constraints are applied
        # Handle no constraint
        reference_size = self.num_species
        target_size = reference_size + self.fill_size
        none_log10_count = self.calculateNoconstraintLog10NumAssignment(reference_size, target_size)
        # Run the simulation
        for _ in range(self.num_iteration):
            reference_network = Network.makeRandomNetworkByReactionType(self.num_reaction,
                self.num_species)
            if is_subnet:
                target_network = Network.fill(reference_network, num_fill_reaction=self.fill_size,
                      num_fill_species=self.fill_size)
            else:
                target_network = Network.makeRandomNetworkByReactionType(target_size, target_size)
            for is_species in [True, False]:
                # Initialize for species or reaction
                if is_species:
                    constraint_cls = SpeciesConstraint
                    default_option_collection = SpeciesConstraintOptionCollection()
                    kwarg = 'species_constraint_option_collection'
                    result_dct = species_result_dct
                else:
                    constraint_cls = ReactionConstraint
                    default_option_collection = ReactionConstraintOptionCollection()
                    kwarg = 'reaction_constraint_option_collection'
                    result_dct = reaction_result_dct
                # Iterate on the options
                for option_collection in default_option_collection.iterator():
                    if option_collection.collection_short_name == cn.NONE:
                        result_dct[option_collection.collection_short_name].append(none_log10_count)
                        continue
                    # Evaluate the constraint on the networks provided
                    reference_constraint = constraint_cls(
                          reference_network.reactant_nmat, reference_network.product_nmat,
                          **{kwarg: option_collection})
                    target_constraint = constraint_cls(
                          target_network.reactant_nmat, target_network.product_nmat,
                          **{kwarg: option_collection})
                    compatibility_collection = reference_constraint.makeCompatibilityCollection(
                          target_constraint).compatibility_collection
                    result_dct[option_collection.collection_short_name].append(
                          compatibility_collection.log10_num_assignment)
        # Construct the dataframe
        species_dimension_result = DimensionResult(dataframe=pd.DataFrame(species_result_dct),
              short_to_long_dct=SpeciesConstraintOptionCollection().short_to_long_dct)
        reaction_dimension_result = DimensionResult(dataframe=pd.DataFrame(reaction_result_dct),
                short_to_long_dct=ReactionConstraintOptionCollection().short_to_long_dct)
        return EvaluateConstraintsResult(
              reference_size=reference_size, 
              target_size=target_size,
              species_dimension_result=species_dimension_result,
              reaction_dimension_result=reaction_dimension_result)
    
    def plotCompareConstraints(self, axs=None, is_plot:bool=True,
              **kwargs)->EvaluateConstraintsResult:
        """Bar plot the results of the comparison of constraints.

        Args:
            axs (list, optional): pair of axes. Defaults to None.
            is_plot (bool, optional): plot the results. Defaults to True.
        """
        if axs is None:
            _, axs = plt.subplots(1, 2, figsize=(10, 5))
        result = self.compareConstraints(**kwargs)
        # Construct the plot
        titles = ['Species', 'Reactions']
        for idx, dimension in enumerate(
              [result.species_dimension_result, result.reaction_dimension_result]):
            ax = axs[idx]
            df = dimension.dataframe
            mean_ser = df.mean(axis=0)
            std_ser = df.std(axis=0)
            mean_ser.plot(kind='bar', ax=ax, legend=False, yerr=std_ser)    
            ax.set_title(titles[idx])
            ax.set_ylabel('log10 number of permutations')
            ax.set_xlabel('constraints')
            label_dct = {k: v[3:].replace("_", " ") for k, v in dimension.short_to_long_dct.items()}
            label_dct[cn.NONE] = 'None'
            label_dct['+'.join(dimension.short_to_long_dct.keys())] = 'All'
            labels = [f"{c}: {label_dct[c]}" for c in df.columns]
            pos_arr = [.99, 0.9]
            for label in labels[1:-1]:
                new_label = label.replace("make ", "")
                new_label = new_label.replace(" matrix", "")
                new_label = new_label.replace("n step", "2 step")
                xpos = pos_arr[0] - 1.01*len(new_label)/100
                ax.text(xpos, pos_arr[1], new_label, transform=ax.transAxes, fontsize=10, ha='center')
                pos_arr[1] -= 0.05
        if is_plot:
            plt.show()
        return result
    
    

if __name__ == '__main__':
    size = 6
    #ConstraintBenchmark.plotConstraintStudy(size, 9*size, num_iteration=50, is_plot=True)
    benchmark = ConstraintBenchmark(6, 6, 20)
    _ = benchmark.plotHeatmap(list(range(4, 22, 2)), list(range(10, 105, 5)), percentile=95,
          is_contains_reference=True, num_iteration=500)