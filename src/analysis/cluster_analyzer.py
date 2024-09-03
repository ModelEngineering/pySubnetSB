'''Analyzes a cluster of models.'''

import analysis.constants as cn  # type: ignore
import sirn.constants as cnn  # type: ignore
from analysis.result_accessor import ResultAccessor  # type: ignore
from analysis.summary_statistics import SummaryStatistics # type: ignore

import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Any, Optional
from zipfile import ZipFile

class ClusterAnalyzer(object):
    def __init__(self,
                 oscillator_dir:str,
                 identity:str=cnn.ID_WEAK,
                 max_num_assignment:int=cnn.MAX_NUM_ASSIGNMENT,
    )->None:
        """
        Args:
            identity: kind of identity: cn.ID_WEAK or cn.ID_STRONG
            max_num_assignment: Maximum number of assignments considred
            oscillator_dirs: List of oscillator directory (optional)
        """
        self.oscillator_dir = oscillator_dir
        self.identity = identity
        self.max_num_assignment = max_num_assignment
        self.is_sirn = is_sirn
        path = ResultAccessor.makeDirPath(oscillator_dir,
                is_strong=is_strong, max_num_assignment=max_num_assignment, is_sirn=is_sirn)
        self.summary_statistics = SummaryStatistics(path)
        self.result_accessor = self.summary_statistics.result_accessor
        self.df = self.result_accessor.df

    def getClustersBySize(self, min_size:Optional[int]=None)->List[List[int]]:
        """
        Finds the indices of clusters with the specified minimum size.
        A cluster is specified by a list of indices in self.df.

        Args:
            min_size (int, optional): If None, provide the largest cluster. Defaults to None.
        
        Returns:
            List[List[int]]: List of clusters with the specified minimum size.
        """
        if min_size is None:
            min_size = self.summary_statistics.cluster_size.max_val
        return [g for g in self.summary_statistics.cluster_dct.values() if len(g) >= min_size]