'''Provides access to a single cluster result file for a single Antimony Directory.'''
"""
Concepts
1. Cluster result file. A *.txt file produced by cluster_builder.py.

TODO
1. Provide the original NetworkCollection as well.
2. Validate: is_indeterminate, processing_time
"""


import analysis.constants as cn  # type: ignore
import sirn.constants as cnn  # type: ignore
from sirn.processed_network_collection import ProcessedNetworkCollection  # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore

import collections
import numpy as np
import os
import pandas as pd # type: ignore
from typing import List, Optional
from zipfile import ZipFile


STRONG = "strong"
WEAK = "weak"

DataFileStructure = collections.namedtuple("DataFileStructure", "antimony_dir identity max_num_assignment")


class ResultAccessor(object):

    def __init__(self, cluster_result_path:str)->None:
        """
        Args:
            cluster_result_path: Path to the directory with analysis results from sirn.ClusterBuilder
        """
        self.cluster_result_path = cluster_result_path
        self.processed_network_collections = self._makeProcessedNetworkCollections()
        datafile_structure = self.parseDirPath()
        self.oscillator_dir = datafile_structure.antimony_dir
        self.identity = datafile_structure.identity
        self.max_num_assignment = datafile_structure.max_num_assignment
        #
        self.processed_network_collection_df, self.processed_network_df = self._makeDataframe()

    def _makeProcessedNetworkCollections(self)->List[ProcessedNetworkCollection]:
        """
        Reads the result of identity matching and creates a list of ProcessedNetworkCollection.

        Returns:
            List[ProcessedNetworkCollection]: List of ProcessedNetworkCollection
        """
        with open(self.cluster_result_path, "r") as fd:
            serialization_strs = fd.readlines()
        #
        processed_nework_collections = []
        for serialization_str in serialization_strs:
            processed_nework_collections.append(ProcessedNetworkCollection.deserialize(serialization_str))
        return processed_nework_collections
    
    def getNetworkCollections(self,
          idx:Optional[int]=None,
          network_name:Optional[str]=None,
          min_size:Optional[int]=1)->List[ProcessedNetworkCollection]:
        """Retrieves the ProcessedNetworkCollection(s) that meet the criteria.
             idx: index of the collection
             network_name: a particular network is in the collection
             min_size: minimum size of the collection
        Only one optional argument is specified

        Args:
            idx (Optional[int], optional): _description_. Defaults to None.
            network_name (Optional[str], optional): _description_. Defaults to None.
            min_size (Optional[int], optional): _description_. Defaults to 1.

        Returns:
            List[ProcessedNetworkCollection]
        """
        num_none = (idx is None) + (network_name is None) + (min_size == 1)
        if num_none > 1:
            raise ValueError("Only one optional argument is allowed")
        if num_none == 0:
            raise ValueError("One optional argument is required")
        #
        if idx is not None:
            return [self.processed_network_collections[idx]]
        if network_name is not None:
            return [c for c in self.processed_network_collections if network_name in c.processed_networks]
        if min_size > 1:  # type: ignore
            return [c for c in self.processed_network_collections if len(c.processed_networks) >= min_size]  # type: ignore
        raise ValueError("Invalid arguments")

    def parseDirPath(self)->DataFileStructure:
        """Extracts information from the directory path.

        Args:
            dir_path (str): _description_

        Returns:
            List[str]: _description_
        """
        filename = os.path.basename(self.cluster_result_path)
        filename = filename[:-4]
        #
        if STRONG in filename:
            identity = cnn.ID_STRONG
            remainder = filename[len(STRONG):]
        elif WEAK in filename:
            identity = cnn.ID_WEAK
            remainder = filename[len(WEAK):]
        else:
            raise ValueError(f"Invalid filename: {filename}")
        #
        idx = remainder.find("_")
        max_num_assignment = int(remainder[:idx])
        antimony_dir = remainder[idx+1:]
        #
        return DataFileStructure(antimony_dir=antimony_dir, identity=identity,
                                 max_num_assignment=max_num_assignment)

    def _makeDataframe(self)->pd.DataFrame:
        """
        Reads the result of identity matching and creates a dataframe.

        Returns:
            pd.DataFrame: DataFrame of ProcessedNetworkCollections
            pd.DataFrame: DataFrame of constituent ProcessedNetworks 
        """
        with open(self.cluster_result_path, "r") as fd:
            repr_strs = fd.readlines()
        #
        processed_network_dct:dict = {c: [] for c in cn.RESULT_ACCESSOR_PROCESSED_NETWORK_COLUMNS}
        processed_network_collection_dct:dict = {c: [] for c in cn.RESULT_ACCESSOR_PROCESSED_NETWORK_COLLECTION_COLUMNS}
        collection_idx = 0
        for processed_network_collection in self.processed_network_collections:
            processed_networks = processed_network_collection.processed_networks
            processed_network_collection_dct[cn.COL_COLLECTION_IDX].append(collection_idx)
            processed_network_collection_dct[cn.COL_NUM_NETWORK].append(len(processed_networks))
            processed_network_collection_dct[cn.COL_HASH].append(processed_network_collection.hash_val)
            for processed_network in processed_networks:
                processed_network_dct[cn.COL_NETWORK_NAME].append(processed_network.network_name)
                processed_network_dct[cn.COL_PROCESSING_TIME].append(processed_network.processing_time)
                processed_network_dct[cn.COL_IS_INDETERMINATE].append(processed_network.is_indeterminate)
                processed_network_dct[cn.COL_LEN_ASSIGNMENT_COLLECTION].append(
                      len(processed_network.assignment_collection))
                processed_network_dct[cn.COL_COLLECTION_IDX].append(collection_idx)
            collection_idx += 1
        processed_network_collection_df = pd.DataFrame(processed_network_collection_dct)
        processed_network_df = pd.DataFrame(processed_network_dct)
        processed_network_collection_df.attrs[cn.META_IS_STRONG] = self.identity
        processed_network_collection_df.attrs[cn.META_MAX_NUM_ASSIGNMENT] = self.max_num_assignment
        processed_network_collection_df.attrs[cn.COL_OSCILLATOR_DIR] = self.oscillator_dir
        return processed_network_collection_df, processed_network_df

    @staticmethod 
    def iterateDir(result_dir:str, root_dir=cn.DATA_DIR):
        """
        Iterates over the directories in the root directory.

        Args:
            result_dir (str): path to the root directory
            root_dir (str, optional): path to the root directory. Defaults to cn.DATA_DIR.

        Returns:
            str: name of directory
            processed_network_collection_df: collection statistics
            processed_network_df: network statistics
        """
        dir_path = os.path.join(root_dir, result_dir)
        ffiles = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        for file in ffiles:
            full_path = os.path.join(dir_path, file)
            accessor = ResultAccessor(full_path)
            yield accessor.oscillator_dir, accessor.processed_network_collection_df, accessor.processed_network_df

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
        missing_dct:dict = {cn.COL_OSCILLATOR_DIR: [], cn.COL_NETWORK_NAME: []}
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
                subset_model_name = subset_df.loc[subset_group[0], cn.COL_NETWORK_NAME]
                superset_collection_idx = superset_df[superset_df[
                      cn.COL_NETWORK_NAME] == subset_model_name][cn.COL_COLLECTION_IDX].values[0]
                # All members of the cluster (group) in subset 
                # should have the same collection_idx in superset
                for idx in subset_group[1:]:
                    model_name = subset_df.loc[idx, cn.COL_NETWORK_NAME]
                    collection_idx = superset_df[superset_df[
                          cn.COL_NETWORK_NAME] == model_name][cn.COL_COLLECTION_IDX].values[0]
                    if collection_idx != superset_collection_idx:
                        missing_dct[cn.COL_OSCILLATOR_DIR].append(antimony_dir)
                        missing_dct[cn.COL_NETWORK_NAME].append(model_name)
        #
        return missing_dct

    def getAntimonyFromNetworkname(self, model_name:str)->str:
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
        network_names = self.processed_network_df[
              self.processed_network_df[cn.COL_COLLECTION_IDX] == collection_idx][cn.COL_NETWORK_NAME]
        antimony_strs = [self.getAntimonyFromNetworkname(n) for n in network_names]
        return antimony_strs
    
    @staticmethod 
    def getClusterResultPath(oscillator_dir:str="", identity:str=cnn.ID_STRONG,
                             max_num_assignment:int=10000)->str:
        """
        Constructs the path to the cluster results.

        Args:
            oscillator_dir (str): Name of the oscillator directory or "" if not specified
            identity: True for strong, False for weak. Defaults to True.
            max_num_assignment (int, optional): Maximum number of permutations. Defaults to 10000.

        Returns:
            str: path to directory with the cluster results
        """
        prefix = cn.STRONG if identity == cnn.ID_STRONG else cn.WEAK
        maxassignment_condition = f"{prefix}{max_num_assignment}"
        filename = f"{maxassignment_condition}_{oscillator_dir}.txt"
        if len(oscillator_dir) == 0:
            return os.path.join(cn.SIRN_DIR, maxassignment_condition)
        else:
            return os.path.join(cn.SIRN_DIR, maxassignment_condition, filename)

    @staticmethod 
    def getNetworkCollectionFromCSVFile(csv_file:str)->ProcessedNetworkCollection:
        """"
        Constructs a network collection a CSV file of serialized collection of Antimony models.

        Args:
            file_path (str): Path to the serialized antimony models

        Returns:
            ProcessedNetworkCollection
        """
        df = pd.read_csv(csv_file)
        df = df.rename(columns={'num_col': cnn.S_NUM_REACTION, 'num_row': cnn.S_NUM_SPECIES,
            'column_names': cnn.S_REACTION_NAMES, 'row_names': cnn.S_SPECIES_NAMES,
            'reactant_array_str': cnn.S_REACTANT_LST, 'product_array_str': cnn.S_PRODUCT_LST,
            'model_name': cnn.S_NETWORK_NAME})
        serialization_str = NetworkCollection.dataframeToJson(df)
        return NetworkCollection.deserialize(serialization_str)