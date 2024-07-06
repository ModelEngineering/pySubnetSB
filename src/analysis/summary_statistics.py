'''Calculates statistics and simple plots'''


import analysis.constants as cn  # type: ignore
import sirn.constants as cnn  # type: ignore
from analysis.result_accessor import ResultAccessor  # type: ignore
from sirn import util # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # type: ignore
from typing import List, Tuple, Any


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
        self.df_groups = self.df.groupby(cn.COL_MODEL_NAME).groups
        #
        self.num_model = len(self.df)
        self.num_perm_statistics = util.calculateSummaryStatistics(self.df[cn.COL_NUM_PERM])
        self.num_indeterminate_statistics = util.calculateSummaryStatistics(
              [1 if v else 0 for v in self.df[cn.COL_IS_INDETERMINATE]])
        self.processing_time_statistics = util.calculateSummaryStatistics(self.df[cn.COL_PROCESSING_TIME])
        self.cluster_statistics = util.calculateSummaryStatistics(
                [len(v) for v in self.df_groups.values()])

    @classmethod
    def plotComparisonBars(cls, measurement_dirs:List[str], metrics:List[str],
            indices:List[str])->Tuple[Any, pd.DataFrame]:
        """
          Does grouped bar plots across all antimony directories. Groups together
          the root_dirs and/or metrics. Only one of root_dirs or metrics can have more
          than one element.

        Args:
            measurement_dirs (List[str]): List of directories with results for comparison,
                one for each directory
            metrics (List[str]): metrics in cn.RESULT_ACCESSOR_COLUMNS. Calculate mean values
            indices: names of the rows (what is being compared)
        """
        # Create the plot DataFrame
        dct:dict = {n: [] for n in cnn.OSCILLATOR_DIRS}
        true_indices = [str(i) for i in indices]
        for measurement_dir in measurement_dirs:
            for metric in metrics:
                iter = ResultAccessor.iterateDir(measurement_dir)
                for oscillator_directory, df in iter:
                    dct[oscillator_directory].append(df[metric].mean())
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
    def plotMetricComparison(cls, metric:str, is_plot=True)->None:
        """
        Plots the maximum number of permutations.

        Args:
            metric: metric to plot ("max_num_perm", "processing_time", "is_indeterminate") 
        
        """
        max_dct = {cn.COL_PROCESSING_TIME: 0.003, cn.COL_NUM_PERM: 200, cn.COL_IS_INDETERMINATE: 1}
        unit_dct = {cn.COL_PROCESSING_TIME: "sec", cn.COL_NUM_PERM: "", cn.COL_IS_INDETERMINATE: ""}
        for identity_type in [cn.WEAK, cn.STRONG]:
            measurement_dirs = [os.path.join(cn.DATA_DIR, "sirn_analysis", f"{identity_type}{n}")
                                for n in cn.MAX_NUM_PERMS]
            ax, _ = cls.plotComparisonBars(measurement_dirs, [metric], cn.MAX_NUM_PERMS)
            ax.set_title(f"{identity_type}")
            if unit_dct[metric] != "":
                unit_str = f" ({unit_dct[metric]})"
            else:
                unit_str = ""
            ax.set_ylabel(f"avg. {metric} per network ({unit_dct[metric]})")
            ax.set_xlabel("Antimony Directory")
            ax.set_ylim([0, max_dct[metric]])
        if is_plot:
            plt.show()