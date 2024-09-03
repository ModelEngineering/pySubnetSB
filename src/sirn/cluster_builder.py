'''Builds clustered networks from a NetworkCollection based on their structural identity.'''

from sirn import constants as cn  # type: ignore
from src.sirn.network import Network  # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore
from sirn.clustered_network import ClusteredNetwork # type: ignore
from sirn.clustered_network_collection import ClusteredNetworkCollection # type: ignore

import numpy as np
import pandas as pd  # type: ignore
import time
from typing import List, Dict


###############################################
class ClusterBuilder(object):
    # Builds ClusterNetworks from a NetworkCollection based on their structural identity

    def __init__(self, network_collection:NetworkCollection, is_report=True,
                 max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT, is_sirn:bool=True,
                 identity:str=cn.ID_WEAK):
        """
        Args:
            network_collection (NetworkCollection): Collection of networks to cluster
            is_report (bool, optional): Progress reporting
            max_num_perm (float, optional): Maximum log10 of the number of permutations that
                are examined
            is_sirn (bool, optional): Whether the SIRN algorithm is used
            is_structural_identity_strong (bool, optional): Criteria for structurally identical
        """
        self.network_collection = network_collection
        self.is_sirn = is_sirn
        self.is_report = is_report # Progress reporting
        self.max_num_assignment = max_num_assignment  # Maximum number of assignments permitted before indeterminate
        self.identity = identity
        # Statistics
        self.hash_dct = self._makeHashDct()
        self.num_hash = len(self.hash_dct)
        self.max_hash = max([len(l) for l in self.hash_dct.values()])  # Most NetworkCollection
        # Results
        self.clustered_network_collections:List['ClusteredNetworkCollection'] = []

    @property
    def num_indeterminant(self)->int:
        count = 0
        for clustered_network_collection in self.clustered_network_collections:
            count += sum([cn.is_indeterminate for cn in clustered_network_collection.clustered_networks])
        return count
    
    @property
    def collection_sizes(self)->List[int]:
        return [len(cnc) for cnc in self.clustered_network_collections]

    @staticmethod
    def sequenceMax(sequence:List[int])->int:
        if len(sequence) == 0:
            return 0
        return max(sequence)
    
    def _makeHashDct(self)->Dict[int, List[Network]]:
        """
        Makes the hash dictionary for the network collection

        Returns:
            Dict[int, List[Network]]: _description_
        """
        def makeDct(attr:str)->Dict[int, List[Network]]:
            # Build the hash dictionary based on the attribute
            hash_dct: Dict[int, List[Network]] = {}
            # Build the hash dictionary
            for network in self.network_collection.networks:
                hash_val = getattr(network, attr)
                if hash_val in hash_dct:
                    hash_dct[hash_val].append(network)
                else:
                    hash_dct[hash_val] = [network]
            return hash_dct
        #
        if self.is_sirn:
            hash_simple_dct = makeDct('weak_hash')
            simple_max = self.sequenceMax([len(networks) for networks in hash_simple_dct.values()])
            hash_nonsimple_dct = makeDct('strong_hash')
            nonsimple_max = self.sequenceMax([len(networks) for networks in hash_nonsimple_dct.values()])
            if simple_max < nonsimple_max:
                hash_dct = hash_simple_dct
            else:
                hash_dct = hash_nonsimple_dct
        else:
            hash_dct = {cn.NON_SIRN_HASH: self.network_collection.networks}
        return hash_dct

    def clustered2Network(self, clustered_network:ClusteredNetwork)->Network:
        result = self.network_collection.network_dct[clustered_network.network_name]
        return result

    # FIXME: perm->assignments; report # assignment_pairs, not #permutations
    def cluster(self)->None:
        """
        Clusters the network in the collection by finding those that have structural identity. ClusteredNetwork
        is a wrapper for a network to provide context for clustering.

        Pseudo code:
        For all hash values
            For network with the hash value
                clustered_network_collections = []
                clustered_network = ClusteredNetwork(network)
                For other_clustered_network in clustered_network_collections
                    If the network is structurally identical to any network with this hash value
                        Add the network to a ClusteredNetworkCollection for that network

        Returns:
            Updates sef.clustered_network_collections
        """
        # Initialize result
        self.clustered_network_collections = []
        if self.is_report:
            print(f"\n**Number of hash values: {self.num_hash}", end="")
        # Construct collections of structurally identical Networks
        for idx, (hash_val, hash_networks) in enumerate(self.hash_dct.items()):
            if self.is_report:
                print(f" {np.round((idx+1)/self.num_hash, 2)}.", end="")
            # No processing time for the first network in a hash
            first_clustered_network = ClusteredNetwork(hash_networks[0].network_name, processing_time=0.0)
            # Create list of new collections for this key of hash_dct
            new_clustered_network_collections =  \
                [ClusteredNetworkCollection([first_clustered_network],
                     identity=self.identity,
                     hash_val=hash_val)]
            # Find structurally identical networks and add to the appropriate ClusteredNetworkCollection,
            # creating new ClusteredNetworkCollections as needed.
            for network in hash_networks[1:]:
                clustered_network = ClusteredNetwork(network)  # Wrapper for clustering networks
                is_any_indeterminate = False
                is_selected = False
                for clustered_network_collection in new_clustered_network_collections:
                    first_clustered_network = clustered_network_collection.clustered_networks[0]
                    first_network = self.clustered2Network(first_clustered_network)
                    result = first_network.isStructurallyIdentical(network,
                            max_num_assignment=self.max_num_assignment,
                            identity=self.identity)
                    if result:
                        clustered_network_collection.add(clustered_network)
                        is_selected = True
                        break
                    if (not result) and result.is_truncated:
                        is_any_indeterminate = True
                # Add statistics to the clustered network
                clustered_network.setProcessingTime()
                if is_selected:
                    clustered_network.setIndeterminate(False)
                    clustered_network.setAssignmentCollection(result.assignment_pairs)
                else:
                    # Not structurally identical to any ClusteredNetworkCollection with this hash.
                    # Create a new ClusteredNetworkCollection for this hash.
                    clustered_network.setIndeterminate(is_any_indeterminate)
                    clustered_network_collection = ClusteredNetworkCollection([clustered_network],
                        identity=self.identity,
                        hash_val=hash_val)
                    new_clustered_network_collections.append(clustered_network_collection)
            self.clustered_network_collections.extend(new_clustered_network_collections)
            if self.is_report:
                print(".", end='')
        if self.is_report:
            print(f"\n . Number of network collections: {len(self.clustered_network_collections)}")

    def serializeClusteredNetworkCollections(self)->pd.DataFrame:
        """
        Serializes the clustering result. Information about the clustering is contained in df.attrs,
        a dict of the form {property_name: property_value}.

        Returns:
            pd.DataFrame: Serialized data
        """
        # Augment with the clustered network information
        values = [str(v) for v in self.clustered_network_collections]
        dct = {"clustered_network_repr": values}
        df = pd.DataFrame(dct)
        # Augment the dataframe
        df.attrs = {cn.STRUCTURAL_IDENTITY: self.identity,
                       cn.NUM_HASH: self.num_hash,
                       cn.MAX_HASH: self.max_hash}
        return df
    
    @classmethod
    def deserializeClusteredNetworkCollections(cls, df:pd.DataFrame)->List[ClusteredNetworkCollection]:
        """
        Deserializes the clustering result.

        Args:
            df (pd.DataFrame): Serialized data

        Returns:
            ClusterBuilder: Deserialized data
        """
        return [ClusteredNetworkCollection.makeFromRepr(repr_str)
                                         for repr_str in df.clustered_network_repr.values]
    
    def makeNetworkFromClusteredNetwork(self, clustered_network:ClusteredNetwork)->Network:
        """
        Makes a Network from a ClusteredNetwork

        Args:
            clustered_network (ClusteredNetwork): _description_

        Returns:
            Network: _description_
        """
        network_name = ClusteredNetwork.convertToNetworkName(clustered_network)
        return self.network_collection.network_dct[network_name]  