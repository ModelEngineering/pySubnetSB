'''A container for structurally identical networks and their statistics. Has a string representation.'''

from src.sirn.network_base import NetworkBase  # type: ignore
from sirn import constants as cn  # type: ignore

import collections
import time
from typing import Union, Optional


Repr = collections.namedtuple('Repr', ['is_indeterminate', 'network_name', 'num_perm',
                                        'processing_time'])
TIME_SEP = "#"



class ClusteredNetwork(object):

    def __init__(self, network_name:Union[NetworkBase, str], is_indeterminate:bool=False, num_assignment:int=0,
                 processing_time:Optional[float]=None):
        self.network_name = str(network_name)
        self.is_indeterminate = is_indeterminate
        self.num_assignment = num_assignment
        if processing_time is None:
            self.processing_time = time.process_time()  # Time to process the network. Initialized to current time.
        else:
            self.processing_time = processing_time

    def finished(self)->None:
        self.processing_time = time.process_time() - self.processing_time
        if self.processing_time < 0:
            raise ValueError(f"Processing time is negative: {self.processing_time}")

    def __eq__(self, other:object)->bool:
        if not isinstance(other, ClusteredNetwork):
            return False
        return (self.network_name == other.network_name and
                self.is_indeterminate == other.is_indeterminate and
                self.num_assignment == other.num_assignment)
    
    def copy(self)->'ClusteredNetwork':
        return ClusteredNetwork(self.network_name, self.is_indeterminate, self.num_assignment)

    def __repr__(self)->str:
        # Indeterminate networks are prefixed with "?"
        if self.is_indeterminate:
            prefix = cn.NETWORK_NAME_PREFIX_UNKNOWN
        else:
            prefix = cn.NETWORK_NAME_PREFIX_KNOWN
        repr_str = f"{prefix}{self.network_name}{cn.NETWORK_NAME_SUFFIX}{self.num_assignment}#{self.processing_time}"
        return repr_str
    
    def add(self, num_perm:int)->None:
        self.num_assignment += num_perm

    @staticmethod
    def parseRepr(repr_str:str)->Repr:
        """
        Parses a string representation of a ClusteredNetwork.

        Args:
            repr_str (str): _description_

        Returns:
            str
        """
        repr_str = repr_str.strip()
        is_indeterminate_str, rest_str = repr_str[0], repr_str[1:]
        split_str = rest_str.split(cn.NETWORK_NAME_SUFFIX)
        if len(split_str) > 3:
            new_split_str = cn.NETWORK_NAME_SUFFIX.join(split_str[:-3]), split_str[-1]
        else:
            new_split_str = split_str # type: ignore
        if len(new_split_str) > 2:
            network_name = cn.NETWORK_NAME_SUFFIX.join(new_split_str[:-1])
        else:
            network_name = new_split_str[0]
        num_perm_str, processing_time_str = new_split_str[-1].split(TIME_SEP)
        is_indeterminate = bool(is_indeterminate_str == cn.NETWORK_NAME_PREFIX_UNKNOWN)
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