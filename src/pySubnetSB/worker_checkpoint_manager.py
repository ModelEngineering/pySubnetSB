'''Worker for running SubnetFinder in parallel.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.checkpoint_manager import CheckpointManager # type: ignore

import collections
import json
import numpy as np
import pandas as pd  # type: ignore
from typing import Tuple, Optional

PruneResult = collections.namedtuple("PruneResult", ["full_df", "pruned_df", "num_truncated",
      "reference_names"])


class WorkerCheckpointManager(CheckpointManager):
    RecoverResult = collections.namedtuple("RecoverResult", ["full_df", "pruned_df", "processeds"])
    MergedCheckpointResult = collections.namedtuple("MergedCheckpointResult",
           ["num_reference_network", "merged_checkpoint_manager", "dataframe",
            "worker_checkpoint_managers"])

    # Specialization of CheckpointManager for executeTask to checkpoint worker results

    def __init__(self, worker_checkpoint_path:str, is_report:bool=True, is_initialize:bool=False)->None:
        """
        Args:
            subnet_finder (SubnetFinder): SubnetFinder instance
            worker_checkpoint_path (str): Path to the CSV file for the worker checkpoint
            is_report (bool): If True, reports progress
        """
        super().__init__(worker_checkpoint_path, is_report=is_report, is_initialize=is_initialize)

    def recover(self)->RecoverResult:
        """
        Recovers a previously saved DataFrame. The recovered dataframe deletes entries with model strings that are null.

        Returns:
            pd.DataFrame: DataFrame of the checkpoint
            pd.DataFrame: DataFrame of the checkpoint stripped of null entries
            np.ndarray: List of processed reference networks
        """
        df = super().recover()
        is_success = False
        if len(df) > 0:
            try:
                full_df = pd.read_csv(self.path)
                is_success = True
            except:
                pass
        if is_success:
            prune_result = WorkerCheckpointManager.prune(full_df)
            pruned_df = prune_result.pruned_df
            reference_names = prune_result.reference_names
            # Convert the JSON string to a dictionary
            if len(pruned_df) > 0:
                pruned_df.loc[:, cn.FINDER_NAME_DCT] = pruned_df[cn.FINDER_NAME_DCT].apply(lambda x: json.loads(x))
        else:
            full_df = pd.DataFrame()
            pruned_df = pd.DataFrame()
            reference_names = []
        self._print(f"Recovering {len(reference_names)} processed models from {self.path}")
        return self.RecoverResult(full_df=full_df, pruned_df=pruned_df, processeds=reference_names)

    @staticmethod 
    def prune(df:pd.DataFrame)->PruneResult:
        """
        Prunes a DataFrame to include only rows where the reference network is not the null string.

        Args:
            df (pd.DataFrame): Table of matching networks
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str): may be the null string
                target_network (str): may be the null string

        Returns: PruneResult
            pd.DataFrame: Pruned DataFrame
            list: List of reference networks that were pruned
            num_truncated (int): Number of entries where search is truncated
            reference_names (list): Names of reference networks processed
        """
        is_null = df[cn.FINDER_REFERENCE_NETWORK].isnull()
        is_null_str = df[cn.FINDER_REFERENCE_NETWORK] == cn.NULL_STR
        not_sel = is_null | is_null_str
        reference_names = list(set(df[not_sel][cn.FINDER_REFERENCE_NAME].values))
        num_truncated = np.sum(df[cn.FINDER_IS_TRUNCATED])
        prune_result = PruneResult(full_df=df, pruned_df=df[~not_sel], num_truncated=num_truncated,
              reference_names=reference_names)
        return prune_result

    @staticmethod
    def makeWorkerCheckpointPath(outpath_base, worker_idx:int)->str:
        """
        Constructs the checkpoint path for a worker

        Args:
            outpath_base (str): Path for the base file
            worker_idx (int): Index of the worker

        Returns:
            str: Path to the checkpoint
        """
        splits = outpath_base.split(".")
        outpath_pat = splits[0] + "_%d." + splits[1]
        return outpath_pat % worker_idx
    
    @classmethod
    def merge(cls, base_checkpoint_path:str, num_worker:int,
        merged_checkpoint_result:Optional[MergedCheckpointResult]=None,
        is_report:bool=True)->MergedCheckpointResult:
        """
        Merges the checkpoints from checkpoint managers. Assumes that worker checkpoint files are named
        with the pattern base_checkpoint_path_%d.csv.

        Args:
            outpath_base (str): Base path for the checkpoint files
            num_worker (int): Number of workers
            merged_checkpoint_result (Optional[MergedCheckpointResult]): Previous Merged checkpoint result
            is_report (bool): If True, reports progress

        Returns: MergedCheckpointResult
            int: Number of merged entries
            CheckpointManager: Checkpoint manager for merged checkpoint
            pd.DataFrame
        """
        if merged_checkpoint_result is None:
            merged_checkpoint_manager = WorkerCheckpointManager(base_checkpoint_path, is_report=is_report)
            worker_checkpoint_managers = [WorkerCheckpointManager(
                    WorkerCheckpointManager.makeWorkerCheckpointPath(base_checkpoint_path, i),
                    is_report=is_report, is_initialize=False)
                    for i in range(num_worker)]
        else:
            merged_checkpoint_manager = merged_checkpoint_result.merged_checkpoint_manager
            worker_checkpoint_managers = merged_checkpoint_result.worker_checkpoint_managers
        recovers = [m.recover().full_df for m in worker_checkpoint_managers]
        is_initialized = False
        if len(recovers) > 0:
            full_df = pd.concat(recovers, ignore_index=True)
            if len(full_df) > 0:
                merged_checkpoint_manager.checkpoint(full_df)
                num_reference_network = len(full_df[cn.FINDER_REFERENCE_NAME].unique())
                is_initialized = True
        if not is_initialized:
            full_df = pd.DataFrame()
            num_reference_network = 0
        result = cls.MergedCheckpointResult(
                num_reference_network=num_reference_network,
                merged_checkpoint_manager=merged_checkpoint_manager,
                worker_checkpoint_managers=worker_checkpoint_managers,
                dataframe=full_df)           
        return result
    
    @classmethod
    def deleteWorkerCheckpoints(cls, base_checkpoint_path:str, num_worker:int, is_report:bool=True)->None:
        """
        Deletes the worker checkpoints

        Args:
            base_checkpoint_path (str): Base path for the checkpoint files
            num_worker (int): Number of workers
        """
        worker_checkpoint_managers = [cls(cls.makeWorkerCheckpointPath(base_checkpoint_path, i),
              is_report=is_report)
              for i in range(num_worker)]
        [m.remove() for m in worker_checkpoint_managers]