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
                 is_strong:bool=True,
                 max_num_perm:int=10000,
                 is_sirn:bool=True,
    )->None:
        """
        Args:
            is_strong: True for strong, False for weak
            max_num_perm: Maximum number of permutations
            is_sirn: True for SIRN, False for naive
            oscillator_dirs: List of oscillator directory (optional)
        """
        self.oscillator_dir = oscillator_dir
        self.is_strong = is_strong
        self.max_num_perm = max_num_perm
        self.is_sirn = is_sirn
        path = ResultAccessor.makeDirPath(oscillator_dir,
                is_strong=is_strong, max_num_perm=max_num_perm, is_sirn=is_sirn)
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