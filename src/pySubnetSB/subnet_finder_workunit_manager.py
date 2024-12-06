'''Manage the distribution of work for SubnetFinder workers.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.network import Network # type: ignore
from pySubnetSB.model_serializer import ModelSerializer # type: ignore

import collections  # type: ignore
import itertools
import pandas as pd  # type: ignore
import numpy as np # type: ignore
import os
from typing import List, Optional, Tuple


Workunit = collections.namedtuple("Workunit", ['reference_idx', 'target_idx', 'worker_idx',
      'reference_name', 'target_name'])


class SubnetFinderWorkunitManager(object):

    def __init__(self, csv_path:str, num_worker:Optional[int]=None,
          reference_networks:Optional[List[Network]]=None,
          target_networks:Optional[List[Network]]=None):
        """
        Args:
            path (str): path to the CSV file used to store the workunits.
            num_worker (Optional[int], optional): Number of workers. Defaults to None.
            reference_networks (Optional[List[Network]], optional): _description_. Defaults to None.
            target_networks (Optional[List[Network]], optional): _description_. Defaults to None.
        """
        self.num_worker = num_worker
        self.reference_networks = reference_networks
        self.target_networks = target_networks
        self.num_reference_network = None if reference_networks is None else len(reference_networks)
        self.num_target_network = None if target_networks is None else len(target_networks)
        self.csv_path = csv_path

    @classmethod
    def makeFromSerializationFiles(cls, csv_path:str, num_worker:int,
          reference_serialization_path:str,
          target_serialization_path:str)->'SubnetFinderWorkunitManager':
        """
        Create a workunit manager from serialization files.

        Args:
            reference_serialization_path (str): Path to the reference networks.
            target_serialization_path (str): Path to the target networks.
        """
        serializer = ModelSerializer(None, reference_serialization_path)
        reference_networks = serializer.deserialize().networks
        serializer = ModelSerializer(None, target_serialization_path)
        target_networks = serializer.deserialize().networks
        return cls(csv_path, num_worker=num_worker,
              reference_networks=reference_networks, target_networks=target_networks)

    def makeWorkunitFile(self, reference_serialization_path:Optional[str]=None,
          target_serialization_path:Optional[str]=None)->None:
        """
        Creates a CSV file of workunits.
        Randomly assigns workers to work units.
        Make workunit pairs for each worker.

        Args:
            reference_serialization_path (Optional[str], optional): Path to the reference networks. Defaults to None.
            target_serialization_path (Optional[str], optional): Path to the target networks. Defaults
        """
        #####
        def getNetworks(serialization_path:Optional[str],
              default_networks:Optional[List[Network]])->Optional[List[Network]]:
            if serialization_path is None:
                return default_networks
            # Deserialize and return the networks
            serializer = ModelSerializer(None, serialization_path)
            return serializer.deserialize().networks
        #####
        reference_networks = getNetworks(reference_serialization_path, self.reference_networks)
        target_networks = getNetworks(target_serialization_path, self.target_networks)
        if reference_networks is None or target_networks is None:
            raise ValueError("Reference or target networks are not set.")
        num_reference_network = len(reference_networks)
        num_target_network = len(target_networks)
        num_workunit_pairs = num_reference_network * num_target_network
        network_idx_pairs = [(i, j) for i, j in itertools.product(
              range(num_reference_network), range(num_target_network))]
        worker_idxs = np.array(range(num_workunit_pairs)) % self.num_worker
        np.random.shuffle(worker_idxs)   # Randomize the assignments
        reference_names = [reference_networks[w[0]].network_name for w in network_idx_pairs]
        target_names = [target_networks[w[1]].network_name for w in network_idx_pairs]
        df = pd.DataFrame({cn.FINDER_REFERENCE_IDX: [network_idx_pairs[i][0] for i in range(len(worker_idxs))],
              cn.FINDER_TARGET_IDX: [network_idx_pairs[i][1] for i in range(len(worker_idxs))],
              cn.FINDER_WORKER_IDX: worker_idxs,
              cn.FINDER_REFERENCE_NAME: reference_names,
              cn.FINDER_TARGET_NAME: target_names})
        df.to_csv(self.csv_path, index=False)

    def getWorkunitDataframe(self, worker_idx:Optional[int]=None)->pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise ValueError(f"File does not exist: {self.csv_path}.")
        df = pd.read_csv(self.csv_path)
        if worker_idx is not None:
            worker_df = df[df[cn.FINDER_WORKER_IDX] == worker_idx]
        else:
            worker_df = df
        return worker_df
    
    def getWorkunits(self, worker_idx:int,
          processed_reference_networks:Optional[List[str]]=None,
          processed_target_networks:Optional[List[str]]=None,
          )->List[Workunit]:
        """
        Get the workunits for a worker, eliminating any workunits that have been processed.

        Args:
            worker_idx (int): index of the worker
            processed_reference_networks (Optional[List[str]], optional): List of names of processed
                reference networks paired with processed_target_networks to define the workunit.
            processed_target_networks (Optional[List[str]], optional): List of names of processed target networks
                paired with processed_reference_networks to define the workunit

        Returns:
            List[Workunit]
        """
        worker_df = self.getWorkunitDataframe(worker_idx=worker_idx)
        if processed_reference_networks is None:
            processed_reference_networks = []
        if processed_target_networks is None:
            processed_target_networks = []
        if len(processed_reference_networks) != len(processed_target_networks):
            raise ValueError("Length of processed reference and target networks must be equal.")
        #
        if len(processed_reference_networks) > 0:
            # Eliminate previously processed workunits
            sel = worker_df[cn.FINDER_REFERENCE_NAME].isin(processed_reference_networks)
            sel = np.logical_and (sel, worker_df[cn.FINDER_TARGET_NAME].isin(processed_target_networks))
            worker_df = worker_df[~sel]   # Remove the processed workunits
        workunits = [Workunit(reference_idx=row[cn.FINDER_REFERENCE_IDX], target_idx=row[cn.FINDER_TARGET_IDX], 
              worker_idx=worker_idx, reference_name=row[cn.FINDER_REFERENCE_NAME],
              target_name=row[cn.FINDER_TARGET_NAME]) for _, row in worker_df.iterrows()]
        return workunits
    
    def remove(self):
        """Remove the workunit file."""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)