'''Builds clustered networks from a NetworkCollection based on their structural identity.'''

from sirn import constants as cn
from sirn.network import Network
from sirn.network_collection import NetworkCollection, SERIALIZATION_NAMES
from sirn.clustered_network import ClusteredNetwork
from sirn.clustered_network_collection import ClusteredNetworkCollection

import numpy as np
import pandas as pd  # type: ignore
from typing import List, Dict

IS_STRUCTURAL_IDENTITY_STRONG = 'is_structural_identity_strong'


###############################################
class ClusterBuilder(object):
    # Builds ClusterNetworks from a NetworkCollection based on their structural identity

    def __init__(self, network_collection:NetworkCollection, is_report=True,
                 max_num_perm:int=cn.MAX_NUM_PERM,
                 is_structural_identity_strong:bool=True):
        """
        Args:
            network_collection (NetworkCollection): Collection of networks to cluster
            is_report (bool, optional): Progress reporting
            max_num_perm (float, optional): Maximum log10 of the number of permutations that
                are examined
            is_structural_identity_strong (bool, optional): Criteria for structurally identical
        """
        self.network_collection = network_collection
        self.is_report = is_report # Progress reporting
        self.max_num_perm = max_num_perm  # Maximum number of permutations to search
        self.is_structural_identity_strong = is_structural_identity_strong
        if self.is_structural_identity_strong:
            self.structural_identity_type = cn.STRUCTURAL_IDENTITY_TYPE_STRONG
        else:
            self.structural_identity_type = cn.STRUCTURAL_IDENTITY_TYPE_WEAK
        # Statistics
        self.hash_dct = self._makeHashDct()
        self.num_hash = len(self.hash_dct)
        self.max_hash = max([len(l) for l in self.hash_dct.values()])  # Most NetworkCollection
        # Results
        self.clustered_network_collections:List['ClusteredNetworkCollection'] = []

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
        hash_simple_dct = makeDct('simple_hash')
        simple_max = self.sequenceMax([len(networks) for networks in hash_simple_dct.values()])
        hash_nonsimple_dct = makeDct('nonsimple_hash')
        nonsimple_max = self.sequenceMax([len(networks) for networks in hash_nonsimple_dct.values()])
        if simple_max < nonsimple_max:
            hash_dct = hash_simple_dct
        else:
            hash_dct = hash_nonsimple_dct
        return hash_dct

    def clustered2Network(self, clustered_network:ClusteredNetwork)->Network:
        result = self.network_collection.network_dct[clustered_network.network_name]
        return result

    def cluster(self)->None:
        """
        Clusters the network in the collection by finding those that have structural identity.

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
            first_clustered_network = ClusteredNetwork(str(hash_networks[0]))
            # Create list of new collections for this key of hash_dct
            new_clustered_network_collections =  \
                [ClusteredNetworkCollection([first_clustered_network],
                     is_structural_identity_strong=self.is_structural_identity_strong,
                     hash_val=hash_val)]
            # Find structurally identical networks and add to the appropriate ClusteredNetworkCollection,
            # creating new ClusteredNetworkCollections as needed.a
            for network in hash_networks[1:]:
                clustered_network = ClusteredNetwork(network)
                for clustered_network_collection in new_clustered_network_collections:
                    selected_clustered_network_collection = None
                    first_clustered_network = clustered_network_collection.clustered_networks[0]
                    first_network = self.clustered2Network(first_clustered_network)
                    adjusted_max_num_perm = self.max_num_perm - clustered_network.num_perm
                    result = first_network.isStructurallyIdentical(network,
                            max_num_perm=adjusted_max_num_perm,
                            is_structural_identity_weak=not self.is_structural_identity_strong)
                    clustered_network.add(result.num_perm)
                    if result.is_structural_identity_strong:
                        selected_clustered_network_collection = clustered_network_collection
                        break
                    if result.is_structural_identity_weak and (not self.is_structural_identity_strong):
                        selected_clustered_network_collection = clustered_network_collection
                        break
                    if clustered_network.num_perm >= self.max_num_perm:
                        clustered_network.is_indeterminate = True
                        break
                # Process the result of the search
                if selected_clustered_network_collection is not None:
                    selected_clustered_network_collection.add(clustered_network)
                else:
                    # Not structurally identical to any ClusteredNetworkCollection with this hash
                    clustered_network_collection = ClusteredNetworkCollection([clustered_network],
                        is_structural_identity_strong=self.is_structural_identity_strong,
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
        df.attrs = {cn.STRUCTURAL_IDENTITY_TYPE_STRONG: self.is_structural_identity_strong,
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