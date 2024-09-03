'''A container for collections of structurally identical networks and their statistics.'''
"""
Has a string representation and can construct from its string representation.
"""


from sirn import constants as cn # type: ignore
from sirn.clustered_network import ClusteredNetwork  # type: ignore
from sirn.assignment_pair import AssignmentPair  # type: ignore

import json
import collections
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Optional

Repr = collections.namedtuple('Repr',
     ['identity', 'hash_val', 'clustered_networks'])


class ClusteredNetworkCollection(object):
    # Collection of networks that are structurally identical

    def __init__(self, clustered_networks:List[ClusteredNetwork],
                 identity:str=cn.ID_WEAK, hash_val:int=-1,
                 antimony_directory:Optional[str]=None):
        self.clustered_networks = clustered_networks  # type: ignore
        self.identity = identity
        self.hash_val = hash_val
        self.antimony_directory = antimony_directory # Directory where the network is stored

    @property
    def processing_time(self)->float:
        return np.sum([c.processing_time for c in self.clustered_networks])

    def copy(self)->'ClusteredNetworkCollection':
        return ClusteredNetworkCollection([cn.copy() for cn in self.clustered_networks],
                identity=self.identity,
                hash_val=self.hash_val)

    def __eq__(self, other:object)->bool:
        if not isinstance(other, ClusteredNetworkCollection):
            return False
        if self.identity != other.identity:
            import pdb; pdb.set_trace()
            return False
        if self.hash_val != other.hash_val:
            import pdb; pdb.set_trace()
            return False
        for net1, net2 in zip(self.clustered_networks, other.clustered_networks):
            if not net1 == net2:
                import pdb; pdb.set_trace()
                return False
        return True
    
    def isSubset(self, other:object)->bool:
        # Is this a subset of other?
        if not isinstance(other, ClusteredNetworkCollection):
            return False
        for this_network in self.clustered_networks:
            is_found = False
            for other_network in other.clustered_networks:
                if this_network == other_network:
                    is_found = True
                    break
            if not is_found:
                return False
        return True

    def __len__(self)->int:
        return len(self.clustered_networks)
    
    def __repr__(self)->str:
        # Summary of the object
        if self.identity == cn.ID_STRONG:
            prefix = cn.IDENTITY_PREFIX_STRONG
        else:
            prefix = cn.IDENTITY_PREFIX_WEAK
        names = [c.network_name for c in self.clustered_networks]
        clustered_networks_str = cn.NETWORK_DELIMITER.join(names)
        result = f"{prefix}{self.processing_time}_{self.hash_val}__{clustered_networks_str}"
        return result

    def add(self, clustered_network:ClusteredNetwork):
        self.clustered_networks.append(clustered_network)

    def serialize(self)->str:
        """Creates a JSON string for the object.

        Returns:
            str
        """
        dct = {cn.S_ID: str(self.__class__),
               cn.S_CLUSTERED_NETWORKS: [c.serialize() for c in self.clustered_networks],
               cn.S_IDENITY: self.identity,
               cn.S_HASH_VAL: int(self.hash_val),  # Cannot serialize numpy.int64
               cn.S_ANTIMONY_DIRECTORY: self.antimony_directory,
        }
        return json.dumps(dct)
    
    @classmethod
    def deserialize(cls, serialization_str)->'ClusteredNetworkCollection':
        """Creates a ClusteredNetworkCollection from a JSON serialization string.

        Args:
            serialization_str

        Returns:
            ClusteredNetworkCollection
        """
        dct = json.loads(serialization_str)
        if not str(cls) in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        identity = dct[cn.S_IDENITY]
        antimony_directory =  dct[cn.S_ANTIMONY_DIRECTORY]
        clustered_networks = [ClusteredNetwork.deserialize(s) for s in dct[cn.S_CLUSTERED_NETWORKS]]
        hash_val = dct[cn.S_HASH_VAL]
        clustered_network_collection = ClusteredNetworkCollection(clustered_networks, identity=identity,
                                          hash_val=hash_val, antimony_directory=antimony_directory)
        return clustered_network_collection