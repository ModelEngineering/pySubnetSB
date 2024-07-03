'''A container for structurally identical networks and their statistics. Has a string representation.'''

from sirn.network import Network  # type: ignore
from sirn import constants as cn  # type: ignore

import collections
import time
from typing import Union, Optional


Repr = collections.namedtuple('Repr', ['is_indeterminate', 'network_name', 'num_perm', 'processing_time'])
TIME_SEP = "#"



class ClusteredNetwork(object):

    def __init__(self, network_name:Union[Network, str], is_indeterminate:bool=False, num_perm:int=0,
                 processing_time:Optional[float]=None):
        self.network_name = str(network_name)
        self.is_indeterminate = is_indeterminate
        self.num_perm = num_perm
        if processing_time is None:
            self.processing_time = time.time()  # Time to process the network. Initialized to current time.
        else:
            self.processing_time = processing_time

    def finished(self)->None:
        self.processing_time = time.time() - self.processing_time
        if self.processing_time < 0:
            raise ValueError(f"Processing time is negative: {self.processing_time}")

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
        repr_str = f"{prefix}{self.network_name}{cn.NETWORK_NAME_SUFFIX}{self.num_perm}#{self.processing_time}"
        return repr_str
    
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
        rest_str = rest_str.split(cn.NETWORK_NAME_SUFFIX)[0]
        num_perm_str, processing_time_str = rest_str.split(TIME_SEP)
        is_indeterminate = is_indeterminate_str == cn.NETWORK_NAME_PREFIX_UNKNOWN
        num_perm = int(num_perm_str)
        return Repr(is_indeterminate=is_indeterminate, network_name=network_name, num_perm=num_perm,
                    processing_time=float(processing_time_str))

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
        clustered_repr = ClusteredNetwork(repr.network_name, repr.is_indeterminate, repr.num_perm)
        clustered_repr.processing_time = repr.processing_time
        return clustered_repr