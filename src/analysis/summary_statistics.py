'''Calculates statistics and simple plots'''

"""
Key concepts:
   Metric. A measurement produced by the cluster analysis (e.g., number of cluster_size)
   Aggregation. A function that summarizes the metric (e.g., mean, total, min_val, max_val)
   Value. The result of applying the aggregation function to the metric. There is one value
            for each metric, oscillator directory, and condition directory.
   Oscillator directory. Directory with the results of the analysis of a single oscillator.
   Condition directory. Directory with analysis results from sirn.ClusterBuilder. There is one
         file for each oscillator directory. Conditions are:
            - WEAK vs. STRONG
            - MAX_NUM_PERM: maximum number of permutations
            - NAIVE algorithm vs. SIRN algorithm
   Groupby. The criteria for grouping together multiple values.
            (e.g., group by the oscillator directory).
"""


import analysis.constants as cn  # type: ignore
import sirn.constants as cnn  # type: ignore
from analysis.result_accessor import ResultAccessor  # type: ignore
from sirn import util # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # type: ignore
from typing import List, Tuple, Any, Optional


class SummaryStatistics(object):

    def __init__(self, analysis_directory:str, root_path:str=cn.DATA_DIR)->None:
        """
        Args:
            analysis_directory: Directory with analysis results from sirn.ClusterBuilder
            root_path: Root directory for the analysis_directory
        """
        self.analysis_directory = analysis_directory
        self.root_path = root_path
        self.result_accessor = ResultAccessor(os.path.join(root_path, analysis_directory))
        self.df = self.result_accessor.df
        self.cluster_dct = self.df.groupby(cn.COL_MODEL_NAME).groups
        #
        self.num_model = util.calculateSummaryStatistics(np.repeat(1, len(self.df)))
        self.num_perm = util.calculateSummaryStatistics(self.df[cn.COL_NUM_PERM])
        self.indeterminate = util.calculateSummaryStatistics(
              [1 if v else 0 for v in self.df[cn.COL_IS_INDETERMINATE]])
        self.processing_time = util.calculateSummaryStatistics(self.df[cn.COL_PROCESSING_TIME])
        # Cluster statistics
        cluster_sizes = [len(v) for v in self.cluster_dct.values()]
        self.cluster_size = util.calculateSummaryStatistics(
                [v for v in cluster_sizes])
        self.cluster_size_eq1 = util.calculateSummaryStatistics(
                [1 if v == 1 else 0 for v in cluster_sizes])
        self.cluster_size_gt1 = util.calculateSummaryStatistics(
                [1 if v > 1 else 0 for v in cluster_sizes])

    @classmethod
    def plotConditionByOscillatorDirectory(cls, condition_dirs:List[str], metric:str,
            indices:List[str])->Tuple[Any, pd.DataFrame]:
        """
          Does grouped bar plots between directories that have the oscillator analysis result files.
          Uses the mean value of the metric.

        Args:
            condition_dirs (List[str]): List of directories with conditions for comparison.
                Each condition directory has a file for each Oscillator Directory
            metric [str]: metrics in cn.RESULT_ACCESSOR_COLUMNS. Calculate mean values
            indices: names of the rows (what is being compared)
        """
        # Create the plot DataFrame
        dct:dict = {n: [] for n in cnn.OSCILLATOR_DIRS}
        true_indices = [str(i) for i in indices]
        for measurement_dir in condition_dirs:
            iter = ResultAccessor.iterateDir(measurement_dir)
            for oscillator_directory, df in iter:
                dct[oscillator_directory].append(df[metric].mean())
        import pdb; pdb.set_trace()
        plot_df = pd.DataFrame(dct, columns=cnn.OSCILLATOR_DIRS, index=true_indices)
        plot_df = plot_df.transpose()
        # Plot
        ax = plot_df.plot.bar()
        labels = []
        for directory in cnn.OSCILLATOR_DIRS:
            label = directory.replace("Oscillators_", "")
            num_underscore = label.count("_")
            if num_underscore > 0:
                pos = 0
                for _ in range(num_underscore):
                    pos = label.find("_", pos+1)
                label = label[:pos]
            labels.append(label)
        ax.set_xticklabels(labels, rotation = -50)
        return ax, plot_df

    @classmethod 
    def plotConditionMetrics(cls, metric:str, max_num_perms:Optional[List[int]]=None,
                             is_plot=True)->None:
        """
        Plots a single metric for the conditions WEAK and STRONG grouped by oscillator directory.

        Args:
            metric: metric to plot ("max_num_perm", "processing_time", "is_indeterminate") 
            max_num_perms: list of maximum number of permutations
            is_plot: True if plot is to be displayed
        """
        if max_num_perms is None:
            max_num_perms = cn.MAX_NUM_PERMS
        #
        max_dct = {cn.COL_PROCESSING_TIME: 0.003, cn.COL_NUM_PERM: 200, cn.COL_IS_INDETERMINATE: 1}
        unit_dct = {cn.COL_PROCESSING_TIME: "sec", cn.COL_NUM_PERM: "", cn.COL_IS_INDETERMINATE: ""}
        for identity_type in [cn.WEAK, cn.STRONG]:
            measurement_dirs = [os.path.join(cn.DATA_DIR, "sirn_analysis", f"{identity_type}{n}")
                                for n in max_num_perms]
            ax, _ = cls.plotConditionByOscillatorDirectory(measurement_dirs, metric, cn.MAX_NUM_PERMS)
            ax.set_title(f"{identity_type}")
            ax.set_ylabel(f"avg. {metric} per network ({unit_dct[metric]})")
            ax.set_xlabel("Antimony Directory")
            ax.set_ylim([0, max_dct[metric]])
        if is_plot:
            plt.show()

    @classmethod
    def plotMetricsByOscillatorDirectory(cls, condition_dir:List[str], metrics:List[str],
            aggregations:List[str],
            indices:List[str])->pd.DataFrame:
        """
        Plot multiple metrics for a single condition grouped by oscillator directory.

        Args:
            condition_dir (str): Directory with results for comparison
            metrics (List[str]): Metric 
            aggregations (List[str]): Aggregation function
            indices: names of the rows (what is being compared)
        """
        # Create the plot DataFrame
        dct:dict = {n: [] for n in cnn.OSCILLATOR_DIRS}
        true_indices = [str(i) for i in indices]
        for idx, metric in enumerate(metrics):
            iter = ResultAccessor.iterateDir(condition_dir)
            for oscillator_directory, df in iter:
                summary_statistic = SummaryStatistics(oscillator_directory)
                statistic = getattr(summary_statistic, metric)
                value = getattr(statistic, aggregations[idx])
                dct[oscillator_directory].append(value)
        plot_df = pd.DataFrame(dct, columns=cnn.OSCILLATOR_DIRS, index=true_indices)
        plot_df = plot_df.transpose()
        # Plot
        ax = plot_df.plot.bar()
        labels = []
        for directory in cnn.OSCILLATOR_DIRS:
            label = directory.replace("Oscillators_", "")
            num_underscore = label.count("_")
            if num_underscore > 0:
                pos = 0
                for _ in range(num_underscore):
                    pos = label.find("_", pos+1)
                label = label[:pos]
            labels.append(label)
        ax.set_xticklabels(labels, rotation = -50)
        return ax, plot_df