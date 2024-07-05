'''Provides access to results of checking for structural identity.'''


import analysis.constants as cn  # type: ignore
from sirn.clustered_network_collection import ClusteredNetworkCollection  # type: ignore
from sirn.clustered_network import ClusteredNetwork  # type: ignore

import collections
import numpy as np
import os
import pandas as pd # type: ignore
from typing import List, Tuple

STRONG = "strong"
WEAK = "weak"
# Dataframe columns
COL_HASH = "hash"
COL_MODEL_NAME = "model_name"
COL_PROCESS_TIME = "process_time"
COL_NUM_PERM = "num_perm"
COL_IS_INDETERMINATE = "is_indeterminate"
COL_COLLECTION_IDX = "collection_idx"
COLUMNS = [COL_HASH, COL_MODEL_NAME, COL_PROCESS_TIME, COL_NUM_PERM,
           COL_IS_INDETERMINATE, COL_COLLECTION_IDX]
# Dataframe metadata
META_IS_STRONG = "is_strong"
META_MAX_NUM_PERM = "max_num_perm"
META_ANTIMONY_DIR = "antimony_dir"

DataFileStructure = collections.namedtuple("DataFileStructure", "antimony_dir is_strong max_num_perm")


class ResultAccessor(object):

    def __init__(self, dir_path:str)->None:
        """
        Args:
            dir_path: Path to the directory with analysis results from sirn.ClusterBuilder
        """
        self.dir_path = dir_path
        datafile_structure = self.parseDirPath()
        self.antimony_dir = datafile_structure.antimony_dir
        self.is_strong = datafile_structure.is_strong
        self.max_num_perm = datafile_structure.max_num_perm
        #
        self.df = self.makeDataframe()

    def parseDirPath(self)->DataFileStructure:
        """_summary_

        Args:
            dir_path (str): _description_

        Returns:
            List[str]: _description_
        """
        filename = os.path.basename(self.dir_path)
        filename = filename[:-4]
        #
        if STRONG in filename:
            is_strong = True
            remainder = filename[len(STRONG):]
        elif WEAK in filename:
            is_strong = False
            remainder = filename[len(WEAK):]
        else:
            raise ValueError(f"Invalid filename: {filename}")
        #
        idx = remainder.find("_")
        max_num_perm = int(remainder[:idx])
        antimony_dir = remainder[idx+1:]
        #
        return DataFileStructure(antimony_dir=antimony_dir, is_strong=is_strong,
                                 max_num_perm=max_num_perm)
    
    def makeDataframe(self)->pd.DataFrame:
        """
        Reads the result of identity matching and creates a dataframe.

        Returns:
            pd.DataFrame (see COLUMNS)
        """
        with open(self.dir_path, "r") as fd:
            repr_strs = fd.readlines()
        #
        dct:dict = {c: [] for c in COLUMNS}
        collection_idx = 0
        for repr_str in repr_strs:
            collection_repr = ClusteredNetworkCollection.parseRepr(repr_str)
            clustered_networks = collection_repr.clustered_networks
            for clustered_network in clustered_networks:
                dct[COL_HASH].append(collection_repr.hash_val)
                dct[COL_MODEL_NAME].append(clustered_network.network_name)
                dct[COL_PROCESS_TIME].append(clustered_network.processing_time)
                dct[COL_NUM_PERM].append(clustered_network.num_perm)
                dct[COL_IS_INDETERMINATE].append(clustered_network.is_indeterminate)
                dct[COL_COLLECTION_IDX].append(collection_idx)
            collection_idx += 1
        df = pd.DataFrame(dct)
        df.attrs[META_IS_STRONG] = self.is_strong
        df.attrs[META_MAX_NUM_PERM] = self.max_num_perm
        df.attrs[META_ANTIMONY_DIR] = self.antimony_dir
        return df