'''Provides access to a single cluster result file for a single Antimony Directory.'''
"""
Concepts
1. Cluster result file. A *.txt file produced by cluster_builder.py.
"""


import analysis.constants as cn  # type: ignore
from sirn.clustered_network_collection import ClusteredNetworkCollection  # type: ignore
from sirn.clustered_network import ClusteredNetwork  # type: ignore

import collections
import numpy as np
import os
import pandas as pd # type: ignore
from typing import List, Optional
from zipfile import ZipFile

WEAK = "weak"
STRONG = "strong"

DataFileStructure = collections.namedtuple("DataFileStructure", "antimony_dir is_strong max_num_perm")


class ResultAccessor(object):

    def __init__(self, cluster_result_path:str)->None:
        """
        Args:
            cluster_result_path: Path to the directory with analysis results from sirn.ClusterBuilder
        """
        self.cluster_result_path = cluster_result_path
        datafile_structure = self.parseDirPath()
        self.oscillator_dir = datafile_structure.antimony_dir
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
        filename = os.path.basename(self.cluster_result_path)
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
        with open(self.cluster_result_path, "r") as fd:
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
        df.attrs[cn.COL_OSCILLATOR_DIR] = self.oscillator_dir
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
            yield accessor.oscillator_dir, accessor.df

    @classmethod
    def isClusterSubset(cls, superset_dir:str, subset_dir:str)->dict:
        """
        Checks that the the clusters in the subset directory are found in the superset directory.

        Args:
            sirn_result_dir (str)
            naive_result_dir (str)

        Returns:
            dict: keys
                oscillator_dir
                model_name (file name)
        """
        missing_dct:dict = {cn.COL_OSCILLATOR_DIR: [], cn.COL_MODEL_NAME: []}
        # Make dataframe pairs
        superset_iter = cls.iterateDir(superset_dir)
        superset_dct = {directory: df for directory, df in superset_iter}
        subset_iter = cls.iterateDir(subset_dir)
        subset_dct = {directory: df for directory, df in subset_iter}
        # Iterate across all Oscillator directories in the path
        for antimony_dir, subset_df in subset_dct.items():
            if not antimony_dir in superset_dct:
                print(f"Missing {antimony_dir} in superset")
                continue
            superset_df = superset_dct[antimony_dir]
            subset_groups = subset_df.groupby(cn.COL_COLLECTION_IDX).groups
            subset_groups = {k: v for k, v in subset_groups.items() if len(v) > 1}
            # Make sure that subset groups are in the same collection in superset
            for _, subset_group in subset_groups.items():
                # Find the collection_idx in superset for the first model in subset
                subset_model_name = subset_df.loc[subset_group[0], cn.COL_MODEL_NAME]
                superset_collection_idx = superset_df[superset_df[
                      cn.COL_MODEL_NAME] == subset_model_name][cn.COL_COLLECTION_IDX].values[0]
                # All members of the cluster (group) in subset 
                # should have the same collection_idx in superset
                for idx in subset_group[1:]:
                    model_name = subset_df.loc[idx, cn.COL_MODEL_NAME]
                    collection_idx = superset_df[superset_df[
                          cn.COL_MODEL_NAME] == model_name][cn.COL_COLLECTION_IDX].values[0]
                    if collection_idx != superset_collection_idx:
                        missing_dct[cn.COL_OSCILLATOR_DIR].append(antimony_dir)
                        missing_dct[cn.COL_MODEL_NAME].append(model_name)
        #
        return missing_dct

    def getAntimonyFromModelname(self, model_name:str)->str:
        """
        Returns a string, which is the content of the Antimony file.

        Args:
            model_name: Name of the model

        Returns:
            str: Antimony file
        """
        with ZipFile(cn.OSCILLATOR_ZIP, 'r') as zip:
            names = zip.namelist()
        candidates = [n for n in names if model_name in n]
        if len(candidates) > 1:
            raise ValueError(f"Model name {model_name} not uniquely found in {self.oscillator_dir}")
        if len(candidates) == 0:
            raise ValueError(f"Model name {model_name} not found in {self.oscillator_dir}")
        antimony_str = ""
        with ZipFile(cn.OSCILLATOR_ZIP, 'r') as zip:
            fd = zip.open(candidates[0])
            antimony_str = fd.read().decode("utf-8")
        return antimony_str

    def getAntimonyFromCollectionidx(self, collection_idx:int)->List[str]:
        """
        Returns one or more antimony files.

        Args:
            collection_idx: Index of the collection

        Returns:
            str: Antimony file
        """
        model_names = self.df[self.df[cn.COL_COLLECTION_IDX] == collection_idx][cn.COL_MODEL_NAME]
        antimony_strs = [self.getAntimonyFromModelname(n) for n in model_names]
        return antimony_strs
    
    @staticmethod 
    def getClusterResultPath(oscillator_dir:str="", is_strong:bool=True,
                             is_sirn=True, max_num_perm:int=10000)->str:
        """
        Constructs the path to the cluster results.

        Args:
            oscillator_dir (str): Name of the oscillator directory or "" if not specified
            is_strong (bool, optional): True for strong, False for weak. Defaults to True.
            is_sirn (bool, optional): True for SIRN, False for naive. Defaults to True.
            max_num_perm (int, optional): Maximum number of permutations. Defaults to 10000.

        Returns:
            str: path to directory with the cluster results

        Usage:
            # Provide a complete path to the cluster results
            ResultAccessor.getClusterResultPath(oscillator_dir="Oscillators_June_10",
                is_strong=True, is_sirn=True, max_num_perm=10000)
            >> "/Users/jlheller/home/Technical/repos/OscillatorDatabase/sirn_analysis/strong10000/Oscillators_June_10.txt"
            # Provide only the condition path
            ResultAccessor.getClusterResultPath("Oscillators_June_10",
                is_strong=True, is_sirn=True, max_num_perm=10000)
            >> "/Users/jlheller/home/Technical/repos/OscillatorDatabase/sirn_analysis/strong10000"
        """
        prefix = cn.STRONG if is_strong else cn.WEAK
        maxperm_condition = f"{prefix}{max_num_perm}"
        parent_dir = cn.SIRN_DIR if is_sirn else cn.NAIVE_DIR
        filename = f"{maxperm_condition}_{oscillator_dir}.txt"
        if len(oscillator_dir) == 0:
            return os.path.join(parent_dir, maxperm_condition)
        else:
            return os.path.join(parent_dir, maxperm_condition, filename)