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
                 is_strong:bool=True,
                 max_num_perm:int=10000,
                 is_sirn:bool=True,
                 oscillator_dirs:Optional[List[str]]=None)->None:
        """
        Args:
            is_strong: True for strong, False for weak
            max_num_perm: Maximum number of permutations
            is_sirn: True for SIRN, False for naive
            oscillator_dirs: List of oscillator directory (optional)
        """
        self.is_strong = is_strong
        self.max_num_perm = max_num_perm
        self.is_sirn = is_sirn
        self.condition_dir = self.getConditionDir()
        self.summary_statistics = SummaryStatistics(self.condition_dir)
        self.result_accessor = self.summary_statistics.result_accessor
        self.df = self.result_accessor.df
        
    def getConditionDir(self)->str:
        """
        Returns:
            str: directory with the analysis results
        """
        prefix = cn.STRONG if self.is_strong else cn.WEAK
        condition = f"{prefix}{self.max_num_perm}"
        root_dir = cn.SIRN_DIR if self.is_sirn else cn.NAIVE_DIR
        return os.path.join(root_dir, condition)