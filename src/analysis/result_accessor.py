'''Provides access to results of checking for structural identity.'''


import analysis.constants as cn  # type: ignore

import os
from typing import List


class ResultAccessor(object):

    def __init__(self, directory:str, is_strong:bool=True, max_num_perm:int=1000,
                 data_dir=cn.DATA_DIR)->None:
        """
        Args:
            directory: Name of the directory with the Antimony files
            is_strong: Use analysis from strong or weak identity
        """
        self.directory = directory
        self.data_dir = data_dir
        self.is_strong = is_strong
        self.max_num_perm = max_num_perm
        #
        self.results = self.makeResultStr()

    def makeResultStr(self)->List[str]:
        """
        Reads the result of identity matching and creates a list of strings.
    
        Args:
            directory: str
    
        Results:
            str
        """
        if self.is_strong:
            strong_prefix = "strong"
        else:
            strong_prefix = "weak"
        filename = f"{strong_prefix}{self.max_num_perm}_{self.directory}.txt"
        path = os.path.join(self.data_dir, filename)
        with open(path, "r") as fd:
            results = fd.readlines()
        results = [r[:-1] for r in results]
        return results