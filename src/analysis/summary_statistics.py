'''Calculates statistics and simple plots'''


import analysis.constants as cn  # type: ignore
from analysis.result_accessor import ResultAccessor  # type: ignore
from sirn import util # type: ignore

import numpy as np
import os
import pandas as pd # type: ignore
from typing import List, Optional



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
        self.process_time_statistics = util.calculateSummaryStatistics(self.df[cn.COL_PROCESS_TIME])
        self.cluster_statistics = util.calculateSummaryStatistics(
                [len(v) for v in self.df_groups.values()])

    def plotComparisonBars(cls, measurement_dirs:List[str], metrics:List[str],
                           legends:Optional[List[str]]=None)->None:
        """
          Does grouped bar plots across all antimony directories. Groups together
          the root_dirs and/or metrics. Only one of root_dirs or metrics can have more
          than one element.

        Args:
            measurement_dirs (List[str]): List of directories with results for comparison,
                one for each directory
            metrics (List[str]): list of metrics to compare
        """