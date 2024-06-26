'''Provides access to results of checking for structural identity.'''


import analysis.constants as cn

from typing import List


class ResultAccessor(object):

    def __init__(self, directory:str, is_strong:bool=True)->None:
        """
        Args:
            directory: Name of the directory with the Antimony files
            is_strong: Use analysis from strong or weak identity
        """
        self.directory = directory
        self.is_strong = is_strong
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
            dct = cn.IDENTITY_STRONG_PATH_DCT
        else:
            dct = cn.IDENTITY_WEAK_PATH_DCT
        path = dct[self.directory]
        with open(path, "r") as fd:
            results = fd.readlines()
        results = [r[:-1] for r in results]
        return results
            

# TESTS
for is_strong in [True, False]:
    accessor = ResultAccessor('Oscillators_May_28_2024_8898', is_strong=is_strong)
    assert(isinstance(accessor.results, list))
    assert(isinstance(accessor.results[0], str))
print("OK!")