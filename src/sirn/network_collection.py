'''Container of networks. Can serialize and deserialize to a DataFrame. Construct from Antimony files. '''


from sirn import constants as cn
from sirn.network import Network   # type: ignore
from sirn.pmatrix import PMatrix   # type: ignore

import collections
import os
import pandas as pd # type: ignore
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
STRUCTURALLY_IDENTICAL_TYPE = 'structurally_identical_type'
SERIALIZATION_NAMES = [MODEL_NAME, REACTANT_ARRAY_STR, PRODUCT_ARRAY_STR, ROW_NAMES,
                       COLUMN_NAMES, NUM_ROW, NUM_COL, NUM_ROW, NUM_COL]

ArrayContext = collections.namedtuple('ArrayContext', "string, num_row, num_column")


####################################
class NetworkCollection(object):
        
    def __init__(self, networks: List[Network], directory:Optional[str]=None)->None:
        """
        Args:
            networks (List[Network]): Networks in the collection
        """
        self.networks = networks
        self.network_dct = {n.network_name: n for n in networks}
        self.directory = directory

    def add(self, network:Network)->None:
        self.networks.append(network)

    def __eq__(self, other:'NetworkCollection')->bool:  # type: ignore
        # Check that collections have networks with the same attribute values
        if len(self) != len(other):
            return False
        # Check the network names
        network1_dct = {n.network_name: n for n in self.networks}
        network2_dct = {n.network_name: n for n in other.networks}
        key_set = set(network1_dct.keys())
        key_diff = key_set.symmetric_difference(set(network2_dct.keys()))
        if len(key_diff) > 0:
            return False
        #
        for key in key_set:
            if not network1_dct[key] == network2_dct[key]:
                return False
        return True

    def __len__(self)->int:
        return len(self.networks)
    
    def __repr__(self)->str:
        names = [str(n) for n in self.networks]
        return "---".join(names)
    
    def _findCommonType(self, collection_identity_type1:str, collection_identity_type2:str)->str:
        if (collection_identity_type1 == cn.STRUCTURAL_IDENTITY_TYPE_NOT) \
                or (collection_identity_type2 == cn.STRUCTURAL_IDENTITY_TYPE_NOT):
            return cn.STRUCTURAL_IDENTITY_TYPE_NOT
        if (collection_identity_type1 == cn.STRUCTURAL_IDENTITY_TYPE_WEAK) \
                or (collection_identity_type2 == cn.STRUCTURAL_IDENTITY_TYPE_WEAK):
            return cn.STRUCTURAL_IDENTITY_TYPE_WEAK
        return cn.STRUCTURAL_IDENTITY_TYPE_STRONG
    
    def __add__(self, other:'NetworkCollection')->'NetworkCollection':
        """
        Union of two network collections. Constructs the correct collection_identity_type.

        Args:
            other (NetworkCollection)

        Returns:
            NetworkCollection: _description_

        Raises:
            ValueError: Common names between collections
        """
        # Error checking
        this_names = set([n.network_name for n in self.networks])
        other_names = set([n.network_name for n in other.networks]) 
        common_names = this_names.intersection(other_names)
        if len(common_names) > 0:
            raise ValueError(f"Common names between collections: {common_names}")
        # Construct the new collection
        networks = self.networks.copy()
        networks.extend(other.networks)
        directory = None
        if self.directory == other.directory:
            directory = self.directory
        return NetworkCollection(networks, directory=directory)
    
    def copy(self)->'NetworkCollection':
        return NetworkCollection(self.networks.copy())
    
    @classmethod
    def makeRandomCollection(cls, array_size:int=3, num_network:int=10,
            structural_identity_type:str=cn.STRUCTURAL_IDENTITY_TYPE_NOT)->'NetworkCollection':
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
            if structural_identity_type:
                network = networks[0].randomize(structural_identity_type=structural_identity_type)
            else:
                network = _make()
            networks.append(network)
        return cls(networks)

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
        return NetworkCollection(networks, directory=indir_path)
    
    @staticmethod 
    def _array2Context(array:np.ndarray)->ArrayContext:
        num_row, num_column = np.shape(array)
        flat_array = np.reshape(array, num_row*num_column)
        str_arr = [str(i) for i in flat_array]
        array_str = "[" + ",".join(str_arr) + "]"
        return ArrayContext(array_str, num_row, num_column)
    
    @staticmethod
    def _string2Array(array_context:ArrayContext)->np.ndarray:
        array = np.array(eval(array_context.string))
        array = np.reshape(array, (array_context.num_row, array_context.num_column))
        return array

    def serialize(self)->pd.DataFrame:
        """Constructs a DataFrame from a NetworkCollection

        Returns:
            pd.DataFrame: See SERIALIZATION_NAMES
               DataFrame.metadata: Dictionary of metadata
                    "directory": Directory of the Antimony files
        """
        dct: Dict[str, list] = {n: [] for n in SERIALIZATION_NAMES}
        for network in self.networks:
            if len(network.reactant_pmatrix.row_names) != network.reactant_pmatrix.num_row:
                raise ValueError("row_names and column_names must have the same length.")
            if len(network.reactant_pmatrix.column_names) != network.reactant_pmatrix.num_column:
                raise ValueError("column_names and column_names must have the same length.")
            dct[MODEL_NAME].append(network.network_name)
            reactant_pmatrix = network.reactant_pmatrix
            product_pmatrix = network.product_pmatrix
            reactant_array_context = self._array2Context(reactant_pmatrix.array)
            dct[REACTANT_ARRAY_STR].append(reactant_array_context.string)
            product_array_context = self._array2Context(product_pmatrix.array)
            dct[PRODUCT_ARRAY_STR].append(product_array_context.string)
            dct[NUM_ROW].append(reactant_array_context.num_row)
            dct[NUM_COL].append(reactant_array_context.num_column)
            dct[ROW_NAMES].append(str(reactant_pmatrix.row_names))  # type: ignore
            dct[COLUMN_NAMES].append(str(reactant_pmatrix.column_names))  # type: ignore
        df = pd.DataFrame(dct)
        df.metadata = {"directory": self.directory}
        return df

    @classmethod 
    def deserialize(cls, df:pd.DataFrame)->'NetworkCollection':
        """Deserializes a DataFrame to a NetworkCollection

        Args:
            df: pd.DataFrame

        Returns:
            PMatrixCollection
        """
        def _makePMatrix(array_str:str, num_row:int, num_column:int, row_names:str,
                         column_names:str):
            array_context = ArrayContext(array_str, num_row, num_column)
            array = cls._string2Array(array_context)
            return PMatrix(array, row_names=eval(row_names), column_names=eval(column_names))
        #
        networks = []
        for _, row in df.iterrows():
            row_names:str = row[ROW_NAMES]  # type: ignore
            column_names:str = row[COLUMN_NAMES]  # type: ignore
            num_row = row[NUM_ROW]
            num_col = row[NUM_COL]
            reactant_array_str = row[REACTANT_ARRAY_STR]
            reactant_pmatrix = _makePMatrix(reactant_array_str,
                                            num_row, num_col, row_names, column_names) 
            product_pmatrix = _makePMatrix(row[PRODUCT_ARRAY_STR],
                                            num_row, num_col, row_names, column_names) 
            network = Network(reactant_pmatrix, product_pmatrix, network_name=row[MODEL_NAME])
            networks.append(network)
        return NetworkCollection(networks)