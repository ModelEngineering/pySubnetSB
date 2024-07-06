'''Provides access to results of checking for structural identity.'''


import analysis.constants as cn  # type: ignore
from sirn.clustered_network_collection import ClusteredNetworkCollection  # type: ignore
from sirn.clustered_network import ClusteredNetwork  # type: ignore

import collections
import numpy as np
import os
import pandas as pd # type: ignore
from typing import Tuple

WEAK = "weak"
STRONG = "strong"

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
            pd.DataFrame (see cn.RESULT_ACCESSOR_COLUMNS)
        """
        with open(self.dir_path, "r") as fd:
            repr_strs = fd.readlines()
        #
        dct:dict = {c: [] for c in cn.RESULT_ACCESSOR_COLUMNS}
        collection_idx = 0
        for repr_str in repr_strs:
            collection_repr = ClusteredNetworkCollection.parseRepr(repr_str)
            clustered_networks = collection_repr.clustered_networks
            for clustered_network in clustered_networks:
                dct[cn.COL_HASH].append(collection_repr.hash_val)
                dct[cn.COL_MODEL_NAME].append(clustered_network.network_name)
                dct[cn.COL_PROCESSING_TIME].append(clustered_network.processing_time)
                dct[cn.COL_NUM_PERM].append(clustered_network.num_perm)
                dct[cn.COL_IS_INDETERMINATE].append(clustered_network.is_indeterminate)
                dct[cn.COL_COLLECTION_IDX].append(collection_idx)
            collection_idx += 1
        df = pd.DataFrame(dct)
        df.attrs[cn.META_IS_STRONG] = self.is_strong
        df.attrs[cn.META_MAX_NUM_PERM] = self.max_num_perm
        df.attrs[cn.META_ANTIMONY_DIR] = self.antimony_dir
        return df

    @staticmethod 
    def iterateDir(result_dir:str, root_dir=cn.DATA_DIR):
        """
        Iterates over the directories in the root directory.

        Args:
            result_dir (str): path to the root directory
            root_dir (str, optional): path to the root directory. Defaults to cn.DATA_DIR.

        Returns:
            Tuple[str, pd.DataFrame]: name of directory, dataframe of statistics
        """
        dir_path = os.path.join(root_dir, result_dir)
        ffiles = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        for file in ffiles:
            full_path = os.path.join(dir_path, file)
            accessor = ResultAccessor(full_path)
            yield accessor.antimony_dir, accessor.df