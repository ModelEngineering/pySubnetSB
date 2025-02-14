'''Benchmarks various aspects of pySubnetSB.'''

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

import scipy.special  # type: ignore
from pySubnetSB.significance_calculator_core import SignificanceCalculatorCore  # type: ignore
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
import tqdm  # type: ignore
import time
from typing import List, Optional, Tuple

NULL_DF = pd.DataFrame()
C_TIME = 'time'
C_LOG10_NUM_PERMUTATION = 'log10num_permutation'
C_NUM_REFERENCE = 'num_reference'
C_NUM_TARGET = 'num_target'
TITLE_FONT_SIZE_INCREMENT = 2
POC_SIGNIFICANCE = "poc_significance"

DimensionResult = collections.namedtuple('DimensionResult', ['dataframe', 'short_to_long_dct'])
EvaluateConstraintsResult = collections.namedtuple('EvaluateConstraintsResult',
      ['reference_size', 'target_size', 'species_dimension_result', 'reaction_dimension_result'])


# A study result is a container of the results of multiple benchmarks
StudyResult = collections.namedtuple('StudyResult', ['is_categorical_id', 'study_ids', 'benchmark_results'])
#  is_categorical_id: bool # True if study_ids are categorical
#  study_ids: List[str | float]
#  benchmark_results: List[pd.DataFrame]


class Benchmark(object):
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
        benchmark = Benchmark(reference_size, fill_size=fill_size, num_iteration=num_iteration)
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
          is_no_constraint:bool=False, title:Optional[str]=None,
          ax=None, font_size:int=8, is_cbar:bool=True, num_digit:int=1)->plt.Axes:
        """Plot a heatmap of the log10 of number of permutations.

        Args:
            num_references (List[int]): number of reference networks
            num_targets (List[int]): number of target networks
            percentile (int): percentile of distribution of the log number of permutation
            is_plot (bool, optional): plot the heatmap. Defaults to True.
            is_not_constraint (bool, optional): True if no constraints are applied. Defaults to False.  
            title (str, optional): title of the plot. Defaults to None.
            ax: Matplotlib axes
            font_size (int, optional): font size of the color bar. Defaults to 8.
            is_cbar (bool, optional): show the color bar. Defaults to True.

        Returns:
            matplotlib.axes._axes.Axes: _description_
        """
        #####
        def _round(value:float, num_digit:int)->float:
            if np.isnan(value):
                return np.nan
            if num_digit == 0:
                result = round(value)
            else:
                result = np.round(value, num_digit)
            return result
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
                        result = _round(np.percentile(df[C_LOG10_NUM_PERMUTATION].values, percentile),
                                num_digit)
                        # Calculate multiple percentiles
                        percentile_10 = _round(np.percentile(df[C_LOG10_NUM_PERMUTATION].values, 10),
                              num_digit)
                        percentile_90 = _round(np.percentile(df[C_LOG10_NUM_PERMUTATION].values, 90),
                              num_digit)
                data_dct[C_LOG10_NUM_PERMUTATION].append(_round(result, num_digit))
                if np.isnan(percentile_10):
                    data_dct[C_LABEL].append("")
                else:
                    if is_no_constraint:
                        data_dct[C_LABEL].append(f"{_round(result, num_digit)}")
                    else:
                        annot = f"{_round(result, 1)}\n({np.round(percentile_10, num_digit)}, "
                        annot += f"{_round(percentile_90, num_digit)})"
                        data_dct[C_LABEL].append(annot)
                data_dct[C_LOG10_P10].append(_round(percentile_10, num_digit))
                data_dct[C_LOG10_P90].append(_round(percentile_90, num_digit))
        # Construct the dataframe
        df = pd.DataFrame(data_dct)
        df = df.rename(columns={C_NUM_REFERENCE: 'Reference', C_NUM_TARGET: 'Target'})
        if percentile > 0:
            df[C_LOG10_NUM_PERMUTATION] = [_round(v, num_digit) for v in df[C_LOG10_NUM_PERMUTATION]]
        else:
            df[C_LOG10_NUM_PERMUTATION] = [_round(v, num_digit)
                  for v in df[C_LOG10_NUM_PERMUTATION].astype(float)]
        pivot_df = df.pivot(columns='Reference', index='Target', values=C_LOG10_NUM_PERMUTATION)
        pivot_df = pivot_df.sort_index(ascending=False)
        label_df = df.pivot(columns='Reference', index='Target', values=C_LABEL)
        label_df = label_df.sort_index(ascending=False)
        # Plot
        ax = sns.heatmap(pivot_df, annot=label_df.values, fmt="", cmap='Reds', vmin=0, vmax=15,
                          annot_kws={'size': font_size}, ax=ax, cbar=is_cbar,
                          cbar_kws={'label': 'log10 number of permutations'})
        ax.figure.axes[-1].yaxis.label.set_size(font_size)
        cbar_ticklabels = ax.figure.axes[-1].get_yticklabels()
        ax.figure.axes[-1].set_yticklabels(cbar_ticklabels, size=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_xlabel('reference size', size=font_size)
        ax.set_ylabel('target size', size=font_size)
        if title is not None:
            ax.set_title(title, size=font_size + TITLE_FONT_SIZE_INCREMENT)
        else:
            ax.set_title(f'percentile: {percentile}th (10th, 90th)',
                  size=font_size + TITLE_FONT_SIZE_INCREMENT)
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
    
    def plotCompareConstraints(self, axs=None, font_size:int=12, is_plot:bool=True,
              **kwargs)->EvaluateConstraintsResult:
        """Bar plot the results of the comparison of constraints.

        Args:
            axs (list, optional): pair of axes. Defaults to None.
            is_plot (bool, optional): plot the results. Defaults to True.
        """
        DCT = {
            'none': 'none',
            'a': '1',
            'b': '2,3',
            'a+b': '1,2,3',
            'y': '1,2',
            'z': '3,4',
            'y+z': '1,2,3,4',
        }
        if axs is None:
            _, axs = plt.subplots(1, 2, figsize=(10, 5))
        result = self.compareConstraints(**kwargs)
        # Construct the plot
        titles = ['Reactions', 'Species']
        for idx, dimension in enumerate(
              [result.species_dimension_result, result.reaction_dimension_result]):
            ax = axs[idx]
            df = dimension.dataframe
            mean_ser = df.mean(axis=0)
            std_ser = df.std(axis=0)
            mean_ser.plot(kind='bar', ax=ax, legend=False, yerr=std_ser)    
            ax.set_title(titles[idx], size=font_size + TITLE_FONT_SIZE_INCREMENT)
            yticklabels = ax.get_yticklabels()
            ax.set_yticklabels(yticklabels, size=font_size)
            if idx == 0:
                ax.set_ylabel('log10 number of assignments', size=font_size)
            ax.set_xlabel('constraints', size=font_size, labelpad=2)
            xticklabels = [DCT[v] for v in list(df.columns)]
            ax.set_xticklabels(xticklabels, size=font_size, rotation=0)
            # Legends
            if False:
                label_dct = {k: v[3:].replace("_", " ") for k, v in dimension.short_to_long_dct.items()}
                label_dct[cn.NONE] = 'None'
                label_dct['+'.join(dimension.short_to_long_dct.keys())] = 'All'
                labels = [f"{c}: {label_dct[c]}" for c in df.columns]
                pos_arr = [.90, 0.9]
                for label in labels[1:-1]:
                    new_label = label.replace("make ", "")
                    new_label = new_label.replace(" matrix", "")
                    new_label = new_label.replace("n step", "2 step")
                    xpos = pos_arr[0] - 1.01*len(new_label)/100
                    ax.text(xpos, pos_arr[1], new_label, transform=ax.transAxes, fontsize=font_size,
                        ha='center')
                    pos_arr[1] -= 0.05
                    ticklabels = ax.get_xticklabels()
                    ax.set_xticklabels(ticklabels, fontsize=font_size, rotation=0)
        if is_plot:
            plt.show()
        return result
    
    @classmethod
    def calculateOccurrence(cls, species_reaction_sizes:List[Tuple[int, int]],
          identity:str=cn.ID_WEAK, max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
          num_replication:int=100, num_iteration:int=1000, is_report:bool=True)->pd.DataFrame:
        """Calculates the probability of occurrence of a reference network in a target network.

        Args:
            species_reaction_sizes (List[Tuple[int, int]]): list of tuples of species and reaction sizes
            max_num_assignment (int, optional): _description_. Defaults to cn.MAX_NUM_ASSIGNMENT.
            num_replication: (int): Number of reference networks evaluated for a single combination
                                    of num_reaction, num_species
            num_iteration: (int): Number of target networks compared with a single reference network

        Returns:
            pd.DataFrame: columns
                num_species: number of species
                num_reaction: number of reactions
                mean probability: probability of occurrence            
                std probability: probability of occurrence
                mean_truncated: probability of truncated occurrence
                poc: -log10(probability of occurrence) rounded to a single digit
        """
        COLUMNS = [cn.D_NUM_SPECIES, cn.D_NUM_REACTION, cn.D_MEAN_PROBABILITY, cn.D_STD_PROBABILITY,
                   cn.D_MEAN_TRUNCATED, POC_SIGNIFICANCE, cn.D_NUM_ITERATION]
        MIN_SIGNIFICANCE = 1e-5
        result_dct:dict = {c: [] for c in COLUMNS}
        for num_species, num_reaction in tqdm.tqdm(species_reaction_sizes,
              desc="pairs", disable=not is_report):
            frac_induceds:list = []
            frac_truncateds:list = []
            for _ in range(num_replication):
                reference_network = Network.makeRandomNetworkByReactionType(num_reaction, num_species)
                calculator = SignificanceCalculatorCore(num_species, num_reaction, num_iteration)
                result = calculator.calculateEqual(reference_network, identity=identity,
                        max_num_assignment=max_num_assignment, is_report=False)
                frac_induceds.append(result.frac_induced)
                frac_truncateds.append(result.frac_truncated)
            result_dct[cn.D_NUM_ITERATION].append(result.num_target_network)
            result_dct[cn.D_NUM_SPECIES].append(num_species)
            result_dct[cn.D_NUM_REACTION].append(num_reaction)
            result_dct[cn.D_MEAN_PROBABILITY].append(np.mean(frac_induceds))
            result_dct[cn.D_STD_PROBABILITY].append(np.std(frac_induceds))
            result_dct[cn.D_MEAN_TRUNCATED].append(np.mean(frac_truncateds))
            mean_induced = max(np.nanmean(frac_induceds), MIN_SIGNIFICANCE)
            if not np.isnan(mean_induced):
                poc = round(-np.log10(mean_induced))
            else:
                poc = np.nan
            result_dct[POC_SIGNIFICANCE].append(poc)
        df = pd.DataFrame(result_dct)
        return df

    @classmethod
    def plotSpeciesReactionHeatmap(cls, 
            dataframe:pd.DataFrame, value_column:str, cbar_title:str="",
            plot_title:str="", ax=None, font_size:int=8, is_cbar:bool=True, vmax:Optional[float]=None,
            is_plot:bool=True)->plt.Axes:
        """Plots a heatmap with the horizontal axis as the number of species and the vertical axis as
            the number of reactions.

        Args:
            dataframe (pd.DataFrame): dataframe with columns num_species, num_reaction, value_column
            cbar_title (str): title of the color bar
            value_column (str): column with the values
            font_size (int, optional): font size of the color bar. Defaults to 8.
            is_cbar (bool, optional): show the color bar. Defaults to True.

        Returns:
            matplotlib.axes._axes.Axes: _description_
        """
        # Construct the plot dataframe
        pivot_df = dataframe.pivot(columns=cn.D_NUM_SPECIES, index=cn.D_NUM_REACTION,
              values=POC_SIGNIFICANCE)
        pivot_df = pivot_df.sort_index(ascending=False)
        # Plot
        if vmax is None:
            vmax = pivot_df.max().max()
        ax = sns.heatmap(pivot_df, annot=True, cmap='Reds', vmin=0, vmax=vmax,
                          annot_kws={'size': font_size}, ax=ax, cbar=is_cbar,
                          cbar_kws={'label': cbar_title})
        ax.figure.axes[-1].yaxis.label.set_size(font_size)
        cbar_ticklabels = ax.figure.axes[-1].get_yticklabels()
        ax.figure.axes[-1].set_yticklabels(cbar_ticklabels, size=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_xlabel('number of species', size=font_size)
        ax.set_ylabel('number of reactions', size=font_size)
        ax.set_title(plot_title, size=font_size + TITLE_FONT_SIZE_INCREMENT)
        if is_plot:
            plt.show()
        #
        return ax
    

if __name__ == '__main__':
    size = 6
    #ConstraintBenchmark.plotConstraintStudy(size, 9*size, num_iteration=50, is_plot=True)
    benchmark = Benchmark(6, 6, 20)
    _ = benchmark.plotHeatmap(list(range(4, 22, 2)), list(range(10, 105, 5)), percentile=95,
          is_contains_reference=True, num_iteration=500)