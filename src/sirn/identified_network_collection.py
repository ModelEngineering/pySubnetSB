'''Exposes information about a collection constructed while evaluating structural identity.'''

from sirn.network_collection import NetworkCollection
from sirn.network import Network

import collections
import numpy as np
from typing import Dict, List, Optional



class IdentifiedNetworkCollection(NetworkCollection):

    def __init__(self, networks:List[Network], process_dct:Dict[str, NetworkCollection._Statistics],
                 is_structural_identity_strong:bool=True):
        """
        Args:
            networks (_type_): _description_
            process_dct (Dict[str, _Statistics]): _description_
            is_structural_identity_strong (bool, optional): _description_. Defaults to True.
        """
        super().__init__(networks)
        self.is_structural_identity_strong = is_structural_identity_strong
        self.process_dct = process_dct
        self._num_indeterminate:Optional[int] = None
        self._num_perm:Optional[np.ndarray[int]] = None

    @property
    def num_inderminate(self):
        """
        Returns:
            int
        """
        if self._num_indeterminate is None:
            self._num_indeterminate = sum([1 for _, s in self.process_dct.items()
                                           if s.is_indeterminate])
        return self._num_indeterminate

    @property 
    def  num_perm(self)->np.ndarray[int]:
        """
        Returns:
            int
        """
        if self._num_perm is None:
            self._num_perm = np.array([s.num_perm_evaluated
                                                      for _, s in self.process_dct.items()])
        return self._num_perm
    
    def add(self, network:Network, is_indeterminate:bool=False,
            num_perm:int=0)->None:
        """
        Args:
            network (Network)
            is_indeterminate (bool, optional): Could not determine if permutably identical
            num_perm (int, optional): Number of permutations
        """
        super().add(network)
        self.process_dct[str(network)] = self._Statistics(is_indeterminate=is_indeterminate,
                                                          num_perm=num_perm)
    