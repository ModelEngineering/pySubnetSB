'''Builds ProcessedNetworks from a NetworkCollection based on their structural identity.'''

from pySubnetSB import constants as cn  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.network_collection import NetworkCollection  # type: ignore
from pySubnetSB.processed_network import ProcessedNetwork # type: ignore
from pySubnetSB.processed_network_collection import ProcessedNetworkCollection # type: ignore

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
        self.processed_network_collections:List[ProcessedNetworkCollection] = []

    @property
    def num_indeterminant(self)->int:
        count = 0
        for processed_network_collection in self.processed_network_collections:
            count += sum([cn.is_indeterminate for cn in processed_network_collection.processed_networks])
        return count
    
    @property
    def collection_sizes(self)->List[int]:
        return [len(cnc) for cnc in self.processed_network_collections]

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
            hash_dct = makeDct('network_hash')
        else:
            hash_dct = {cn.OTHER_HASH: self.network_collection.networks}
        return hash_dct

    def processed2Network(self, processed_network:ProcessedNetwork)->Network:
        result = self.network_collection.network_dct[processed_network.network_name]
        return result

    def cluster(self)->None:
        """
        Clusters the network in the collection by finding those that have structural identity. ProcessedNetwork
        is a wrapper for a network to provide context for clustering.

        Pseudo code:
        For all hash values
            For network with the hash value
                processed_network_collections = []
                processed_network = ProcessedNetwork(network)
                For other_processed_network in processed_network_collections
                    If the network is structurally identical to any network with this hash value
                        Add the network to a ProcessedNetworkCollection for that network

        Returns:
            Updates sef.processed_network_collections
        """
        # Initialize result
        self.processed_network_collections = []
        #print(f"\n**Number of hash values: {self.num_hash}", end="")
        # Construct collections of structurally identical Networks
        for idx, (hash_val, hash_networks) in enumerate(self.hash_dct.items()):
            if self.is_report:
                print(f" {np.round((idx+1)/self.num_hash, 2)}.", end="")
            # No processing time for the first network in a hash
            first_processed_network = ProcessedNetwork(hash_networks[0].network_name)
            # Create list of new collections for this key of hash_dct
            new_processed_network_collections =  \
                [ProcessedNetworkCollection([first_processed_network],
                     identity=self.identity,
                     hash_val=hash_val)]
            # Find structurally identical networks and add to the appropriate ProcessedNetworkCollection,
            # creating new ProcessedNetworkCollections as needed.
            for network in hash_networks[1:]:
                processed_network = ProcessedNetwork(network)  # Wrapper for clustering networks
                is_any_indeterminate = False
                is_selected = False
                start_time = time.process_time()
                for processed_network_collection in new_processed_network_collections:
                    first_processed_network = processed_network_collection.processed_networks[0]
                    first_network = self.processed2Network(first_processed_network)
                    result = first_network.isStructurallyIdentical(network,
                            max_num_assignment=self.max_num_assignment,
                            identity=self.identity, is_subnet=False, is_report=self.is_report)
                    if result:
                        processed_network_collection.add(processed_network)
                        is_selected = True
                        break
                    if (not result) and result.is_truncated:
                        is_any_indeterminate = True
                # Add statistics to the ProcessedNetwork
                processed_network.addProcessingTime(time.process_time() - start_time)
                if is_selected:
                    processed_network.setIndeterminate(False)
                    processed_network.setAssignmentCollection(result.assignment_pairs)
                else:
                    # Not structurally identical to any ProcessedNetworkCollection with this hash.
                    # Create a new ProcessedNetworkCollection for this hash.
                    processed_network.setIndeterminate(is_any_indeterminate)
                    processed_network_collection = ProcessedNetworkCollection([processed_network],
                        identity=self.identity,
                        hash_val=hash_val)
                    new_processed_network_collections.append(processed_network_collection)
            self.processed_network_collections.extend(new_processed_network_collections)
            if self.is_report:
                print(".", end='')
        if self.is_report:
            print(f"\n . Number of network collections: {len(self.processed_network_collections)}")

    def serializeProcessedNetworkCollections(self)->pd.DataFrame:
        """
        Serializes the clustering result. Information about the clustering is contained in df.attrs,
        a dict of the form {property_name: property_value}.

        Returns:
            pd.DataFrame: Serialized data
        """
        # Augment with the ProcessedNetwork information
        values = [str(v) for v in self.processed_network_collections]
        dct = {"processed_network_repr": values}
        df = pd.DataFrame(dct)
        # Augment the dataframe
        df.attrs = {cn.STRUCTURAL_IDENTITY: self.identity,
                       cn.NUM_HASH: self.num_hash,
                       cn.MAX_HASH: self.max_hash}
        return df
    
    @classmethod
    def deserializeProcessedNetworkCollections(cls, df:pd.DataFrame)->List[ProcessedNetworkCollection]:
        """
        Deserializes the clustering result.

        Args:
            df (pd.DataFrame): Serialized data

        Returns:
            ClusterBuilder: Deserialized data
        """
        return [ProcessedNetworkCollection.makeFromRepr(repr_str)
                                         for repr_str in df.processed_network_repr.values]
    
    def makeNetworkFromProcessedNetwork(self, processed_network:ProcessedNetwork)->Network:
        """
        Makes a Network from a ProcessedNetwork

        Args:
            processed_network (ProcessedNetwork): _description_

        Returns:
            Network: _description_
        """
        network_name = ProcessedNetwork.convertToNetworkName(processed_network)
        return self.network_collection.network_dct[network_name]  