'''A container for structurally identical networks and their statistics. Has a string representation.'''

from sirn.network import Network
from sirn import constants as cn

import collections
from typing import Union, Tuple


Repr = collections.namedtuple('Repr', ['is_indeterminate', 'network_name', 'num_perm'])



class ClusteredNetwork(object):

    def __init__(self, network_name:Union[Network, str], is_indeterminate:bool=False, num_perm:int=0):
        self.network_name = str(network_name)
        self.is_indeterminate = is_indeterminate
        self.num_perm = num_perm

    def __eq__(self, other:object)->bool:
        if not isinstance(other, ClusteredNetwork):
            return False
        return (self.network_name == other.network_name and
                self.is_indeterminate == other.is_indeterminate and
                self.num_perm == other.num_perm)
    
    def copy(self)->'ClusteredNetwork':
        return ClusteredNetwork(self.network_name, self.is_indeterminate, self.num_perm)

    def __repr__(self)->str:
        # Indeterminate networks are prefixed with "?"
        if self.is_indeterminate:
            prefix = cn.NETWORK_NAME_PREFIX_UNKNOWN
        else:
            prefix = cn.NETWORK_NAME_PREFIX_KNOWN
        return f"{prefix}{self.network_name}{cn.NETWORK_NAME_SUFFIX}{self.num_perm}"
    
    def add(self, num_perm:int)->None:
        self.num_perm += num_perm

    @staticmethod
    def parseRepr(repr_str:str)->Repr:
        """
        Parses a string representation of a ClusteredNetwork.

        Args:
            repr_str (str): _description_

        Returns:
            str
        """
        is_indeterminate_str, rest_str = repr_str[0], repr_str[1:]
        network_name, rest_str = rest_str.split(cn.NETWORK_NAME_SUFFIX)
        num_perm_str = rest_str.split(cn.NETWORK_NAME_SUFFIX)[0]
        is_indeterminate = is_indeterminate_str == cn.NETWORK_NAME_PREFIX_UNKNOWN
        num_perm = int(num_perm_str)
        return Repr(is_indeterminate=is_indeterminate, network_name=network_name, num_perm=num_perm)

    @classmethod
    def makeFromRepr(cls, repr_str:str)->'ClusteredNetwork':
        """
        Constructs a ClusteredNetwork from a string representation.

        Args:
            repr_str (str): _description_

        Returns:
            ClusteredNetwork
        """
        repr = cls.parseRepr(repr_str)
        return ClusteredNetwork(repr.network_name, repr.is_indeterminate, repr.num_perm)