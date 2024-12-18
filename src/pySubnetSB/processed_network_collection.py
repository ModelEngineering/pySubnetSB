'''A container for collections of structurally identical networks and their statistics.'''
"""
Has a string representation and can construct from its string representation.
"""


from pySubnetSB import constants as cn # type: ignore
from pySubnetSB.processed_network import ProcessedNetwork  # type: ignore
from pySubnetSB.assignment_pair import AssignmentPair  # type: ignore

import json
import collections
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Optional

Repr = collections.namedtuple('Repr',
     ['identity', 'hash_val', 'processed_networks'])


class ProcessedNetworkCollection(object):
    # Collection of networks that are structurally identical

    def __init__(self, processed_networks:List[ProcessedNetwork],
                 identity:str=cn.ID_WEAK, hash_val:int=-1,
                 antimony_directory:Optional[str]=None):
        self.processed_networks = processed_networks  # type: ignore
        self.identity = identity
        self.hash_val = hash_val
        self.antimony_directory = antimony_directory # Directory where the network is stored

    @property
    def is_indeterminate(self)->bool:
        return any([n.is_indeterminate for n in self.processed_networks])

    @property
    def processing_time(self)->float:
        return np.sum([c.processing_time for c in self.processed_networks])

    def copy(self)->'ProcessedNetworkCollection':
        return ProcessedNetworkCollection([cn.copy() for cn in self.processed_networks],
                identity=self.identity,
                hash_val=self.hash_val)

    def __eq__(self, other:object)->bool:
        if not isinstance(other, ProcessedNetworkCollection):
            return False
        if self.identity != other.identity:
            import pdb; pdb.set_trace()
            return False
        if self.hash_val != other.hash_val:
            import pdb; pdb.set_trace()
            return False
        for net1, net2 in zip(self.processed_networks, other.processed_networks):
            if not net1 == net2:
                import pdb; pdb.set_trace()
                return False
        return True
    
    def isSubset(self, other:object)->bool:
        # Is this a subset of other?
        if not isinstance(other, ProcessedNetworkCollection):
            return False
        for this_network in self.processed_networks:
            is_found = False
            for other_network in other.processed_networks:
                if this_network == other_network:
                    is_found = True
                    break
            if not is_found:
                return False
        return True

    def __len__(self)->int:
        return len(self.processed_networks)
    
    def __repr__(self)->str:
        # Summary of the object
        if self.identity == cn.ID_STRONG:
            prefix = cn.IDENTITY_PREFIX_STRONG
        else:
            prefix = cn.IDENTITY_PREFIX_WEAK
        names = [c.network_name for c in self.processed_networks]
        processed_networks_str = cn.NETWORK_DELIMITER.join(names)
        result = f"{prefix}{self.processing_time}--{self.hash_val}--{processed_networks_str}"
        return result

    def add(self, processed_network:ProcessedNetwork):
        self.processed_networks.append(processed_network)

    def serialize(self)->str:
        """Creates a JSON string for the object.

        Returns:
            str
        """
        dct = {cn.S_ID: str(self.__class__),
               cn.S_PROCESSED_NETWORKS: [c.serialize() for c in self.processed_networks],
               cn.S_IDENTITY: self.identity,
               cn.S_HASH_VAL: int(self.hash_val),  # Cannot serialize numpy.int64
               cn.S_ANTIMONY_DIRECTORY: self.antimony_directory,
        }
        return json.dumps(dct)
    
    @classmethod
    def deserialize(cls, serialization_str)->'ProcessedNetworkCollection':
        """Creates a ProcessedNetworkCollection from a JSON serialization string.

        Args:
            serialization_str

        Returns:
            ProcessedNetworkCollection
        """
        dct = json.loads(serialization_str)
        if not cls.__name__ in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        identity = dct[cn.S_IDENTITY]
        antimony_directory =  dct[cn.S_ANTIMONY_DIRECTORY]
        processed_networks = [ProcessedNetwork.deserialize(s) for s in dct[cn.S_PROCESSED_NETWORKS]]
        hash_val = dct[cn.S_HASH_VAL]
        processed_network_collection = ProcessedNetworkCollection(processed_networks, identity=identity,
                                          hash_val=hash_val, antimony_directory=antimony_directory)
        return processed_network_collection