
from sirn import constants as cnn
from analysis.result_accessor import ResultAccessor
from analysis import constants as cn

import collections
import numpy as np  # type: ignore
from typing import List, Dict

AggregateStatistics = collections.namedtuple("AggregateStatistics",
        "directories dct")
  # directories: list of strs
  # dct: key is str (name of a statistic); value is list of statistic values

FRC_DUPLICATE_STRONG = "frc_duplicate_strong"
FRC_INDETERMINATE_STRONG = "frc_INDETERMINATE_strong"
TOTAL = "total"
FRC_DUPLICATE_WEAK = "frc_duplicate_weak"
FRC_INDETERMINATE_WEAK = "frc_INDETERMINATE_weak"
STATISTIC_NAMES = [FRC_DUPLICATE_STRONG, FRC_DUPLICATE_WEAK, FRC_INDETERMINATE_STRONG, FRC_INDETERMINATE_WEAK,
                   TOTAL]

class CollectionStatistics(object):
    # Provides statistics about results
    def __init__(self, directory:str, is_strong:bool=True):
        self.directory = directory
        self.is_strong = is_strong
        #
        self.accessor = ResultAccessor(self.directory, is_strong=self.is_strong)
        # Statistics
        num_duplicate = np.sum([v.count("---") for v in self.accessor.results])
        self.total = len(self.accessor.results) + num_duplicate
        self.frc_duplicate = np.sum([v.count("---") for v in self.accessor.results])/self.total
        self.frc_indeterminate = np.sum([v.count("*") for v in self.accessor.results])/self.total
    
    def aggregate(self)->AggregateStatistics:
        """
        Aggregates statistics across the directories
        """
        directories = list(cnn.OSCILLATOR_DIRS)
        dct: Dict[str, List[int]] = {n: [] for n in STATISTIC_NAMES}
        for directory in directories:
            for is_strong in [True, False]:
                accessor = CollectionStatistics(directory, is_strong=is_strong)
                if is_strong:
                    dct[FRC_DUPLICATE_STRONG] = accessor.frc_duplicate
                    dct[FRC_INDETERMINATE_STRONG] = accessor.frc_indeterminate
                else:
                    dct[FRC_DUPLICATE_WEAK] = accessor.frc_duplicate
                    dct[FRC_INDETERMINATE_WEAK] = accessor.frc_indeterminate
                dct[TOTAL] = accessor.total
        return AggregateStatistics(directories=directories, dct=dct)