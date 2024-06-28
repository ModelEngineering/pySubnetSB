'''Builds clustered networks from a NetworkCollection based on their structural identity.'''

from sirn import constants as cn
from sirn.network import Network
from sirn.network_collection import NetworkCollection
from sirn.identified_network_collection import IdentifiedNetworkCollection

from typing import List, Dict, Optional


###############################################
class ClusteredNetwork(object):

    def __init__(self, network:Network, is_indeterminate:bool=False, num_perm:int=0):
        self.network = network
        self.is_indeterminate = is_indeterminate
        self.num_perm = num_perm

    def __repr__(self)->str:
        # Indeterminate networks are prefixed with "?"
        if self.is_indeterminate:
            prefix = "?"
        else:
            prefix = ""
        return f"{prefix}{str(self.network)}"
    
    def add(self, num_perm:int)->None:
        self.num_perm += num_perm
    
    

###############################################
class ClusteredNetworkCollection(object):
    # Collection of networks that are structurally identical
    CSV_HEADER_STR = "network_names, num_network, total_perm"

    def __init__(self, clustered_networks:List[ClusteredNetwork], is_structural_identity_strong:bool=True):
        self.clustered_networks = clustered_networks
        self.is_structural_identity_strong = is_structural_identity_strong
        # Statistics

    def __len__(self)->int:
        return len(self.clustered_networks)
    
    def __repr__(self)->str:
        # Elision of network names, number of networks, and total number of permutation: CSV_HEADER_STR
        if self.is_structural_identity_strong:
            prefix = "+"
        else:
            prefix = "-"
        names = [str(n) for n in self.clustered_networks]
        name_str = cn.NETWORK_NAME_DELIMITER.join(names)
        total_perm = sum([n.num_perm for n in self.clustered_networks]) 
        return f"{prefix}{name_str}, {len(self), {total_perm}}"
    
    def add(self, clustered_network:ClusteredNetwork)->None:
        self.clustered_networks.append(clustered_network)
    


###############################################
class ClusterBuilder(object):
    # Builds ClusterNetworks from a NetworkCollection based on their structural identity

    def __init__(self, network_collection:NetworkCollection, is_report=True,
                 max_log_perm:float=cn.MAX_LOG_PERM,
                 is_structural_identity_strong:bool=True):
        """
        Args:
            network_collection (NetworkCollection): Collection of networks to cluster
            is_report (bool, optional): Progress reporting
            max_log_perm (float, optional): Maximum log10 of the number of permutations that
                are examined
            is_structural_identity_strong (bool, optional): Criteria for structurally identical
        """
        self.network_collection = network_collection
        self.is_report = is_report # Progress reporting
        self.max_log_perm = max_log_perm  # Maximum log of the number of permutations to search
        self.is_structural_identity_strong = is_structural_identity_strong
        if self.is_structural_identity_strong:
            self.structural_identity_type = cn.STRUCTURAL_IDENTITY_TYPE_STRONG
        else:
            self.structural_identity_type = cn.STRUCTURAL_IDENTITY_TYPE_WEAK
        # Results
        self.hash_dct = self._makeHashDct()
        self.num_hash = len(self.hash_dct)
        self.clustered_network_collections:Optional[List[IdentifiedNetworkCollection]] = None

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
        def makeDct(self, attr:str)->Dict[int, List[Network]]:
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
        hash_simple_dct = self.makeDct('simple_hash')
        simple_max = self.sequenceMax([len(networks) for networks in hash_simple_dct.values()])
        hash_nonsimple_dct = self.makeDct('nonsimple_hash')
        nonsimple_max = self.sequenceMax([len(networks) for networks in hash_nonsimple_dct.values()])
        if simple_max < nonsimple_max:
            hash_dct = hash_simple_dct
        else:
            hash_dct = hash_nonsimple_dct
        return hash_dct

    def cluster(self)->None:
        """
        Clusters the network in the collection by finding those that have structural identity.

        Returns:
            Updates sef.clustered_network_collections
        """
        self.clustered_network_collections = []
        # Construct the collections of structurally identical Networks
        for idx, hash_networks in enumerate(self.hash_dct.values()):
            if self.is_report:
                print(f" {idx+1}/{len(hash_networks)}.", end="")
            first_clustered_network = ClusteredNetwork(hash_networks[0])
            clustered_network_collections = [first_clustered_network]
            # Find structurally identical networks and add to the appropriate ClusteredNetworkCollection,
            # creating new ClusteredNetworkCollections as needed.
            for network in hash_networks[1:]:
                clustered_network = ClusteredNetwork(network)
                for clustered_network_collection in clustered_network_collections:
                    selected_clustered_network_collection = None
                    first_network = clustered_network_collection.clustered_networks[0]
                    result = first_network.isStructurallyIdentical(network,
                            max_log_perm=self.max_log_perm,
                            is_structural_identity_weak=not self.is_structural_identity_strong)
                    self.clustered_network.add(result.num_perm)
                    if result.is_structural_identity_strong:
                        selected_clustered_network_collection = clustered_network_collection
                        break
                    if result.is_structural_identity_weak and (not self.is_structural_identity_strong):
                        selected_clustered_network_collection = clustered_network_collection
                        break
                    if self.clustered_network.num_perm > 10**self.max_log_perm:
                        break
                # Process the result of the search
                if selected_clustered_network_collection is not None:
                    # Add to the current IdentifiedNetworkCollection
                    selected_clustered_network_collection.add(clustered_network)
                else:
                    # Create a new IdentifiedNetworkCollection
                    clustered_network_collection = ClusteredNetworkCollection([clustered_network],
                        is_structural_identity_strong=self.is_structural_identity_strong)
                    clustered_network_collections.append(clustered_network_collection)
            self.clustered_network_collections.extend(clustered_network_collections)
            if self.is_report:
                print(".", end='')
        if self.is_report:
            print(f"**Number of network collections: {len(identified_network_collections)}")