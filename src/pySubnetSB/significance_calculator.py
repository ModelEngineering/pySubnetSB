'''Calculates the significance level of finding an induced subnetwork in a target network.'''

"""
The null distribution are randomly generated models tuned to the BioModels repository.
A monte carlo simulation is used to generate the null distribution, and test models to see if
a reference network is present in the target network.

TODO:
1. Automatically choose target sizes to evaluate.
"""

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.significance_calculator_core import  SignificanceCalculatorCore, SignificanceCalculatorCoreResult # type: ignore
from pySubnetSB.network import Network  # type: ignore

import collections
import matplotlib.pyplot as plt
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns # type: ignore
from typing import List, Optional, Tuple  # type: ignore
import tqdm # type: ignore


NUM_REFERENCE_SPECIES = "num_reference_species"
NUM_REFERENCE_REACTION = "num_reference_reaction"
NUM_TARGET_SPECIES = "num_target_species"
NUM_TARGET_REACTION = "num_target_reaction"
FRAC_INDUCED = "frac_induced"
FRAC_TRUNCATED = "frac_truncated"
# Default values
NUM_ITERATION = 1000


PlotSignificanceResult = collections.namedtuple("PlotSignificanceResult",
      ["target_sizes", "frac_induces", "frac_truncates"])   


class SignificanceCalculator(object):

    def __init__(self, reference_network:Network, num_target_species:int, num_target_reaction:int,
                 identity:str=cn.ID_WEAK)->None:
        self.reference_network = reference_network
        self.num_target_reaction = num_target_reaction
        self.num_target_species = num_target_species
        self.identity = identity
    
    def calculate(self, num_iteration:int, is_report:bool=True,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->SignificanceCalculatorCoreResult:
        """
        Calculates the significance level of finding an induced subnetwork in a target network.

        Args:
            num_iteration (int): Number of iterations
            is_report (bool): If True, report progress
            max_num_assignment (int): Maximum number of assignment pairs

        Returns:
            SignificanceCalculatorResult
        """
        core_calculator = SignificanceCalculatorCore(self.num_target_species, self.num_target_reaction,
              num_target_network=num_iteration)
        return core_calculator.calculateSubnet(self.reference_network, identity=self.identity,
                max_num_assignment=max_num_assignment, is_report=is_report)
    
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
                result = calculator.calculate(num_iteration, max_num_assignment=max_num_assignment)
                dct[REFERENCE_SIZE].append(reference_size)
                dct[TARGET_SIZE].append(target_size)
                dct[FRAC_INDUCED].append(result.frac_induced)
                dct[FRAC_TRUNCATED].append(result.frac_truncated)
        # Create the DataFrame
        df = pd.DataFrame(dct)
        return df

    def plotSignificance(self, target_sizes:Optional[List[int]]=None, num_iteration:int=NUM_ITERATION,
          max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
          is_report:bool=True, is_plot=True)->PlotSignificanceResult:
        """
        Plots fraction of random targets in which the reference is used.
        1 - fraction_induced is significance level.

        Args:
            target_sizes (List[int]): _description_
            num_iteration (int, optional): _description_. Defaults to 1000.
            is_report (bool, optional): Report progress. Defaults to True.
            is_plot (bool, optional): Plot

        Returns:
            PlotSignificanceResult

        To further adorn the plot, use is_plot=False and then plt.gca() to get the axis.
        """
        # Set default target sizes
        if target_sizes is None:
            min_size = max(self.reference_network.num_species, self.reference_network.num_reaction)
            target_sizes = [n for n in range(min_size, 6*min_size, min_size)]
        frac_induces = []
        frac_truncates = []
        for target_size in tqdm.tqdm(target_sizes, desc="targets", disable=not is_report):
            calculator = SignificanceCalculator(self.reference_network, target_size,
                target_size, identity=self.identity)
            result = calculator.calculate(num_iteration, max_num_assignment=max_num_assignment,
                is_report=False)
            frac_induces.append(result.frac_induced)
            frac_truncates.append(result.frac_truncated)
        _ = plt.bar(target_sizes, frac_induces, color='blue', alpha=0.5)
        _ = plt.bar(target_sizes, frac_truncates, bottom=frac_induces, color='red', alpha=0.5)
        plt.ylim([0, 1])
        plt.xlabel("Target size")
        plt.ylabel("fraction induced")
        plt.legend(["induced", "truncated"])
        if is_plot:
            plt.show()
        return PlotSignificanceResult(target_sizes=target_sizes, frac_induces=frac_induces,
            frac_truncates=frac_truncates)

    @classmethod 
    def calculateNetworkOccurrenceProbability(cls, reference_network:Network, num_iteration:int=10000,
              identity:str=cn.ID_STRONG, is_report:bool=True,
              max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->Tuple[float, float]:
        """
        Calculates the probability of finding an induced subnetwork in a target network.

        Args:
            reference_network (Network): Reference network
            num_iteration (int): Number of iterations

        Returns:
            float: fraction_induced, fraction_truncated
        """
        """ calculator = SignificanceCalculator(reference_network, reference_network.num_species,
              reference_network.num_reaction)
        result = calculator.calculate(num_iteration, is_report=False)
        return result.frac_induced, result.frac_truncated """
        initial_networks = [Network.makeRandomNetworkByReactionType(reference_network.num_reaction,
              reference_network.num_species, is_exact=True)
             for _ in range(num_iteration)]
        networks = [n for n in initial_networks if n.network_hash == reference_network.network_hash]
        # Calculate the fraction of identical networks for those with the same hash
        num_identical = 0
        num_truncated = 0
        for network in networks:
            result = reference_network.isStructurallyIdentical(network, identity=identity,
                max_num_assignment=max_num_assignment,
                is_subnet=False, is_report=is_report)
            num_identical += 1 if result else 0
            num_truncated += 1 if result.is_truncated else 0
        return num_identical/num_iteration, num_truncated/num_iteration

    @classmethod 
    def plotProbabilityOfOccurrence(cls, data_df:pd.DataFrame, column:str,
          is_plot:bool=True)->None:
        """
        Plots mean and standard deviation (in parens) of the probability of occurrence
        of reaction networks in -log10 units.
          x - number of species
          y - number of reactions
          heatmap - probability of occurrence

        Args:
            data_df (pd.DataFrame): Dataframe with columns num_species
            column (str): Column to analyze. Must be a probability.
        """
        #####
        def transform(df:pd.DataFrame)->pd.DataFrame:
            df.sort_index(level=0, ascending=False, inplace=True)
            df.style.format({c: '{:.2f}' for c in df.columns})
            return df
        #####
        # Transform the data
        df = data_df.copy()
        df[column] = df[column].map(lambda x: max(x, 1e-5))
        df[column] = df[column].map(lambda x: -np.log10(x))
        mean_df = df.pivot_table(index=cn.D_NUM_SPECIES, columns=cn.D_NUM_REACTION,
              values=column, aggfunc='median')
        mean_df = transform(mean_df)
        std_df = df.pivot_table(index=cn.D_NUM_SPECIES, columns=cn.D_NUM_REACTION,
              values=column, aggfunc='std')
        std_df = transform(std_df)
        # Create the labels
        label_df = mean_df.copy().astype(str)
        for index in mean_df.index:
            for column in mean_df.columns:
                is_assigned = False
                if not np.isnan(mean_df.loc[index, column]):
                    if index in std_df.index and column in std_df.columns:
                        text = f"{mean_df.loc[index, column]:1.2f}\n({std_df.loc[index, column]:1.2f})"
                        text = text.replace("(nan)", "")
                        label_df.loc[index, column] = text
                        is_assigned = True
                if not is_assigned:
                    label_df.loc[index, column] = ""
        ax = sns.heatmap(mean_df, annot=label_df.values, fmt="", cmap="coolwarm", vmin=0, vmax=5,
              annot_kws={'size': 8},
              cbar_kws={'label': '-log10 Probability of occurrence'})
        if is_plot:
            plt.show()