'''Calculates statistics and simple plots'''

"""
Key concepts:
   Metric. A measurement produced by the cluster analysis (e.g., number of cluster_size)
   Aggregation. A function that summarizes the metric (e.g., mean, total, min, max)
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

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # type: ignore
from typing import List, Tuple, Any, Optional

CLUSTER_SIZE = "cluster_size"
CLUSTER_SIZE_EQ1 = "cluster_size_eq1"
CLUSTER_SIZE_GT1 = "cluster_size_gt1"

class SummaryStatistics(object):

    def __init__(self, sub_dir:str, data_dir:str=cn.DATA_DIR)->None:
        """
        cluster_result_path = data_dir/condition_dir/oscillator_dir and so
        sub_dir = condition_dir/oscillator_dir

        Args:
            sub_dir: Directory with analysis results from sirn.ClusterBuilder
            data_dir: Root directory for the analysis_directory
        """
        self.sub_dir = sub_dir
        self.data_dir = data_dir
        self.result_accessor = ResultAccessor(os.path.join(data_dir, sub_dir))
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
        self.series = self.makeSeries()

    def makeSeries(self)->pd.DataFrame:
        dct = {}
        def addColumns(base_column, statistics):
            AGGREGATIONS = ["mean", "total", "count", "min", "max"]
            for aggregation in AGGREGATIONS:
                column = f"{base_column}_{aggregation}"
                dct[column] = getattr(statistics, aggregation)
        # Create statistics for the Accessor DataFrame
        accessor_df = self.result_accessor.df
        processing_time = util.calculateSummaryStatistics(accessor_df[cn.COL_PROCESSING_TIME])
        num_perm = util.calculateSummaryStatistics(accessor_df[cn.COL_NUM_PERM])
        is_indeterminate = util.calculateSummaryStatistics(accessor_df[cn.COL_IS_INDETERMINATE])
        collection_idx = util.calculateSummaryStatistics(accessor_df[cn.COL_COLLECTION_IDX])
        base_dct = {CLUSTER_SIZE: self.cluster_size,
                    CLUSTER_SIZE_EQ1: self.cluster_size_eq1,
                    CLUSTER_SIZE_GT1: self.cluster_size_gt1,
                    cn.COL_PROCESSING_TIME: processing_time,
                    cn.COL_NUM_PERM: num_perm,
                    cn.COL_IS_INDETERMINATE: is_indeterminate,
                    cn.COL_COLLECTION_IDX: collection_idx}
        for base_column in base_dct.keys():
            addColumns(base_column, base_dct[base_column])
        #
        ser = pd.Series(dct)
        for key, value in self.result_accessor.df.attrs.items():
            ser.attrs[key] = value
        return ser

    @classmethod
    def iterateOverOscillatorDirectories(cls, condition_dir:str, data_dir:str=cn.DATA_DIR):
        """
        Iterates across the oscillator directories in the condition directory. Provides
        the statistics pd.Series for each oscillator directory.

        Args:
            condition_dir (List[str]): Directories that describe the conditions for analysis
            data_dir (str, optional): _description_. Defaults to cn.DATA_DIR.

        Yields
            pd.Series
        """
        dir_path = os.path.join(data_dir, condition_dir)
        ffiles = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        for file in ffiles:
            full_path = os.path.join(dir_path, file)
            statistics = cls(full_path)
            yield statistics.series

    # FIXME: Deprecate?
    @classmethod
    def plotConditionByOscillatorDirectory(cls, condition_dirs:List[str], metric:str,
            max_num_perms:List[int],
            oscillator_dirs:List[str]=cnn.OSCILLATOR_DIRS,
            title:Optional[str]=None):
        """
          Does grouped bar plots between directories that have the oscillator analysis result files.
          Uses the mean value of the metric.

        Args:
            condition_dirs (List[str]): List of directories with conditions for comparison.
                Each condition directory has a file for each Oscillator Directory
            metric [str]: metrics in cn.STATISTICS_COLUMNS. Calculate mean values
            max_num_perms: maximum number of permutations
        """
        # Create the plot DataFrame
        max_num_perm_strs = [str(i) for i in max_num_perms]
        # Construct a dictionary whose key are oscillator directories and the
        # values correspond to the condition_dirs
        dct:dict = {}
        for condition_dir in condition_dirs:
            iter = cls.iterateOverOscillatorDirectories(condition_dir)
            for ser in iter:
                oscillator_dir = ser.attrs[cn.COL_OSCILLATOR_DIR]
                if not oscillator_dir in dct.keys():
                    dct[oscillator_dir] = [] 
                dct[oscillator_dir].append(ser[metric].values[0])
        plot_df = pd.DataFrame(dct, columns=oscillator_dirs, index=max_num_perm_strs)
        plot_df = plot_df.transpose()
        # Plot the data
        ax = plot_df.plot.bar()
        labels = []
        for directory in oscillator_dirs:
            label = directory.replace("Oscillators_", "")
            num_underscore = label.count("_")
            if num_underscore > 0:
                pos = 0
                for _ in range(num_underscore):
                    pos = label.find("_", pos+1)
                label = label[:pos]
            labels.append(label)
        ax.set_xticklabels(labels, rotation = -50)
        if title is not None:
            ax.set_title(title)
        return ax

    @classmethod 
    def plotMetricByConditions(cls, metric:str,
                             identity_types:List[bool]=[False, True],
                             max_num_perms:List[int]=cn.MAX_NUM_PERMS,
                             sirn_types:List[bool]=[False, True],
                             oscillator_dirs:List[str]=cnn.OSCILLATOR_DIRS,
                             is_log:bool=False,
                             is_plot=True,
                             ylim:Optional[List[float]]=None,
                             legends:Optional[List[str]]=None,
                             **barplot_opts)->Tuple[plt.Axes, pd.DataFrame]:
        """
        Plots a single metric for combinations of conditions:
            - identity_types: weak, strong
            - max_num_perms: maximum number of permutations
            - sirn_types: naive, sirn
        Can restrict the oscillator directories to a subset.

        Args:
            metric: metric to plot ("max_num_perm", "processing_time", "is_indeterminate") 
            max_num_perms: list of maximum number of permutations
            sirn_types: List[boolean]
            identity_types: List of is_strong
            oscillator_dirs: List of oscillator directories
            is_log: True if y-axis is log scale
            ylim: y-axis limits
            is_plot: True if plot is to be displayed
            barplot_opts: options for the bar plot
        """
        # Create the DataFrame to plot
        dct:dict = {n: [] for n in oscillator_dirs}
        labels = []
        for is_strong, is_sirn, max_num_perm in itertools.product(identity_types, sirn_types, max_num_perms):
            condition_dir = ResultAccessor.getClusterResultPath(is_strong=is_strong,
                    max_num_perm=max_num_perm, is_sirn=is_sirn)
            iter = cls.iterateOverOscillatorDirectories(condition_dir)
            strong_label = cn.STRONG if is_strong else cn.WEAK
            sirn_label = cn.SIRN if is_sirn else cn.NAIVE
            label = f"{strong_label}_{max_num_perm}_{sirn_label}"
            labels.append(label)
            for ser in iter:
                oscillator_directory = ser.attrs[cn.COL_OSCILLATOR_DIR]
                if oscillator_directory in oscillator_dirs:
                    value = ser[metric]
                    if is_log:
                        value = np.log10(value)
                    dct[oscillator_directory].append(value)
        plot_df = pd.DataFrame(dct, columns=oscillator_dirs, index=labels)
        plot_df = plot_df.transpose()
        # Do the plot
        ax = plot_df.plot.bar(**barplot_opts)
        concise_dirs = cls._cleanOscillatorLabels(oscillator_dirs)
        ax.set_xticklabels(concise_dirs, rotation = -50)
        # y axis units
        unit = ""
        if is_log:
            unit = "log10"
        if "time" in metric:
            unit += " sec"
        if len(unit) > 0:
            unit = f" ({unit})"
        ax.set_ylabel(f"{metric} per network {unit}")
        ax.set_xlabel("Oscillator Directory")
        if legends is not None:
            ax.legend(legends)
        if ylim is not None:
            ax.set_ylim(ylim)
        if is_plot:
            plt.show()
        return ax, plot_df

    # FIXME: Deprecate?
    @classmethod
    def plotMetricsByOscillatorDirectory(cls, condition_dir:str, metrics:List[str],
            data_dir:str=cn.DATA_DIR)->pd.DataFrame:
        """
        Plot multiple metrics for a single condition grouped by oscillator directory.

        Args:
            condition_dir (str): Directory with results for comparison
            metrics (List[str]): Metric 
            indices: names of the rows (what is being compared)
        """
        # Create the plot DataFrame
        iter = cls.iterateOverOscillatorDirectories(condition_dir)
        dct:dict = {}
        for ser in iter:
            oscillator_directory = ser.attrs[cn.COL_OSCILLATOR_DIR]
            for metric in metrics:
                if not oscillator_directory in dct.keys():
                    dct[oscillator_directory] = []
                dct[oscillator_directory].append(ser[metric])
        #
        plot_df = pd.DataFrame(dct, columns=cnn.OSCILLATOR_DIRS, index=metrics)
        plot_df = plot_df.transpose()
        # Plot
        ax = plot_df.plot.bar()
        labels = cls._cleanOscillatorLabels(cnn.OSCILLATOR_DIRS)
        ax.set_xticklabels(labels, rotation = -50)
        return ax, plot_df
    
    @staticmethod
    def _cleanOscillatorLabels(oscillator_dirs):
        new_labels = []
        for oscillator_dir in oscillator_dirs:
            label = oscillator_dir.replace("Oscillators_", "")
            num_underscore = label.count("_")
            if num_underscore > 0:
                pos = 0
                for _ in range(num_underscore):
                    pos = label.find("_", pos+1)
                label = label[:pos]
            new_labels.append(label)
        return new_labels