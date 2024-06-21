'''Analysis of a collection of Network.'''
"""

A key operation on a collection is cluster. 
The cluster operation groups constructs a collection of network_collection
each of which contains permutably identical matrices.

Notes
1. Handle is_simple_stoichiometry
"""


from sirn.network import Network   # type: ignore
from sirn.pmatrix import PMatrix   # type: ignore

import collections
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict


ANTIMONY_EXTS = [".ant", ".txt", ""]  # Antimony file extensions
MODEL_NAME = 'model_name'
REACTANT_ARRAY_STR = 'reactant_array_str'
PRODUCT_ARRAY_STR = 'product_array_str'
NUM_ROW = 'num_row'
NUM_COL = 'num_col'
ROW_NAMES = 'row_names'
COLUMN_NAMES = 'column_names'
SERIALIZATION_NAMES = [MODEL_NAME, REACTANT_ARRAY_STR, PRODUCT_ARRAY_STR, ROW_NAMES,
                       COLUMN_NAMES, NUM_ROW, NUM_COL, NUM_ROW, NUM_COL]

ArrayContext = collections.namedtuple('ArrayContext', "string, nrow, ncol")


####################################
class NetworkCollection(object):
        
    def __init__(self, networks: List[Network], is_structurally_identical:bool=False,
                 is_simple_stoichiometry:bool=False)->None:
        """
        Args:
            networks (List[Network]): Networks in the collection
            is_structurally_identical (bool, optional): Indicates that networks are structurally identical
            is_simple_stoichiometry (bool, optional): Indicates that structural identical is calculated
                using only the stoichiometry matrix.
        """
        self.networks = networks
        self.is_structurally_identical = is_structurally_identical
        self.is_simple_stoichiometry = is_simple_stoichiometry


    def __len__(self)->int:
        return len(self.networks)
    
    def __repr__(self)->str:
        names = [n.network_name for n in self.networks]
        return "---".join(names)
    
    def __add__(self, other:'NetworkCollection')->'NetworkCollection':
        """
        Union of two network collections.

        Args:
            other (NetworkCollection)

        Returns:
            NetworkCollection: _description_
        """
        network_collection = self.copy()
        is_structurally_identical = self.is_structurally_identical and other.is_structurally_identical
        if is_structurally_identical:
            is_structurally_identical = self.networks[0].isStructurallyIdentical(other.networks[0])
        network_collection.is_structurally_identical = is_structurally_identical
        network_collection.networks.extend(other.networks)
        return network_collection
    
    def copy(self)->'NetworkCollection':
        return NetworkCollection(self.networks.copy(),
                                 is_structurally_identical=self.is_structurally_identical,
                                 is_simple_stoichiometry=self.is_simple_stoichiometry)
    
    @classmethod
    def makeRandomCollection(cls, array_size:int=3, num_network:int=10,
                             is_structurally_identical:bool=False)->'NetworkCollection':
        """
        Make a collection of random networks according to the specified parameters.

        Args:
            array_size (int, optional): Size of the square matrix
            num_network (int, optional): Number of networks
            is_structurally_identical (bool, optional): _description_

        Returns:
            NetworkCollection
        """
        def _make():
            reactant_mat = np.random.randint(-1, 2, (array_size, array_size))
            product_mat = np.random.randint(-1, 2, (array_size, array_size))
            return Network(reactant_mat, product_mat)
        #
        networks = [_make()]
        for _ in range(num_network-1):
            if is_structurally_identical:
                network = networks[0].randomize(is_structurally_identical=True)
            else:
                network = _make()
            networks.append(network)
        return cls(networks, is_structurally_identical=is_structurally_identical,
                   is_simple_stoichiometry=False)
    
    def cluster(self, is_report=True, is_simple_stoichiometry:bool=False)->List['NetworkCollection']:
        """
        Clusters the network in the collection by finding those that are permutably identical.
        Uses the is_simple_stoichiometry flag from the constructor

        Args:
            is_report (bool, optional): _description_
            is_simple_stoichiometry (bool, optional): Criteria for structurally identical

        Returns:
            List[NetworkCollection]: A list of network collections. 
                    Each collection contains network that are permutably identical.
        """
        hash_dct: Dict[int, List[Network]] = {}  # dict values are lists of network with the same hash value
        network_collections = []  # list of permutably identical network collections
        # Build the hash dictionary
        for network in self.networks:
            if network.nonsimple_hash in hash_dct:
                hash_dct[network.nonsimple_hash].append(network)
            else:
                hash_dct[network.nonsimple_hash] = [network]
        if is_report:
            print(f"**Number of hash values: {len(hash_dct)}")
        # Construct the collections of permutably identical matrices
        for idx, networks in enumerate(hash_dct.values()):  # Iterate over collections of pmatrice with the same hash value
            # Find collections of structurally identical networks
            first_collection = [networks[0]]
            new_collections = [first_collection]  # list of collections of permutably identical matrices
            for network in networks[1:]:
                is_in_existing_collection = False
                for new_collection in new_collections:
                    if new_collection[0].isStructurallyIdentical(network,
                            is_simple_stoichiometry=is_simple_stoichiometry): 
                        new_collection.append(network)
                        is_in_existing_collection = True
                        break
                if not is_in_existing_collection:
                    new_collections.append([network])
            if is_report:
                print(".", end='')
            new_network_collections = [NetworkCollection(networks, is_structurally_identical=True) 
                                for networks in new_collections]
            network_collections.extend(new_network_collections)
        return network_collections

    @classmethod
    def makeFromAntimonyDirectory(cls, indir_path:str, max_file:Optional[int]=None,
                processed_network_names:Optional[List[str]]=None,
                report_interval:Optional[int]=None)->'NetworkCollection':
        """Creates a NetworkCollection from a directory of Antimony files.

        Args:
            indir_path (str): Path to the antimony model directory
            max_file (int): Maximum number of files to process
            processed_model_names (List[str]): Names of models already processed
            report_interval (int): Report interval

        Returns:
            NetworkCollection
        """
        ffiles = os.listdir(indir_path)
        networks = []
        network_names = []
        if processed_network_names is not None:
            network_names = list(processed_network_names)
        for count, ffile in enumerate(ffiles):
            if report_interval is not None and count % report_interval == 0:
                is_report = True
            else:
                is_report = False
            network_name = ffile.split('.')[0]
            if network_name in network_names:
                if is_report:
                    print(".", end='')
                continue
            if (max_file is not None) and (count >= max_file):
                break
            if not any([ffile.endswith(ext) for ext in ANTIMONY_EXTS]):
                continue
            path = os.path.join(indir_path, ffile)
            network = Network.makeFromAntimonyFile(path, network_name=network_name)
            networks.append(network)
            if is_report:
                print(f"Processed {count} files.")
        return NetworkCollection(networks)
    
    @staticmethod 
    def _array2Context(array:np.ndarray)->ArrayContext:
        nrow, ncol = np.shape(array)
        flat_array = np.reshape(array, nrow*ncol)
        str_arr = [str(i) for i in flat_array]
        array_str = "[" + ",".join(str_arr) + "]"
        return ArrayContext(array_str, nrow, ncol)
    
    @staticmethod
    def _string2Array(array_context:ArrayContext)->np.ndarray:
        array = np.array(eval(array_context.string))
        array = np.reshape(array, (array_context.nrow, array_context.ncol))
        return array

    def serialize(self)->pd.DataFrame:
        """Constructs a DataFrame from a NetworkCollection

        Returns:
            pd.DataFrame: See SERIALIZATION_NAMES
        """
        dct: Dict[str, list] = {n: [] for n in SERIALIZATION_NAMES}
        for network in self.networks:
            dct[MODEL_NAME].append(network.network_name)
            reactant_pmatrix = network.reactant_pmatrix
            product_pmatrix = network.product_pmatrix
            reactant_array_context = self._array2Context(reactant_pmatrix.array)
            dct[REACTANT_ARRAY_STR].append(reactant_array_context.string)
            product_array_context = self._array2Context(product_pmatrix.array)
            dct[PRODUCT_ARRAY_STR].append(product_array_context.string)
            dct[NUM_ROW].append(reactant_array_context.nrow)
            dct[NUM_COL].append(reactant_array_context.ncol)
            dct[ROW_NAMES].append(str(reactant_pmatrix.row_names))
            dct[COLUMN_NAMES].append(str(reactant_pmatrix.column_names))
        return pd.DataFrame(dct)
    
    @classmethod 
    def deserialize(cls, df:pd.DataFrame)->'NetworkCollection':
        """Deserializes a DataFrame to a NetworkCollection

        Args:
            df: pd.DataFrame

        Returns:
            PMatrixCollection
        """
        def _makePMatrix(array_str:str, num_row:int, num_col:int, row_names:str,
                         column_names:str):
            array_context = ArrayContext(array_str, num_row, num_col)
            array = cls._string2Array(array_context)
            return PMatrix(array, row_names=eval(row_names), column_names=eval(column_names))
        #
        networks = []
        for _, row in df.iterrows():
            row_names:str = row[ROW_NAMES],
            column_names:str = row[COLUMN_NAMES],
            num_row = row[NUM_ROW]
            num_col = row[NUM_COL]
            reactant_pmatrix = _makePMatrix(row[REACTANT_ARRAY_STR],
                                            num_row, num_col, row_names, column_names) 
            product_pmatrix = _makePMatrix(row[PRODUCT_ARRAY_STR],
                                            num_row, num_col, row_names, column_names) 
            pmatrix = Network(reactant_pmatrix, product_pmatrix, network_name=row[MODEL_NAME])
            networks.append(pmatrix)
        return NetworkCollection(networks)