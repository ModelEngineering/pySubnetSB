'''Worker for running SubnetFinder in parallel.'''

import sirn.constants as cn # type: ignore
from sirn.checkpoint_manager import CheckpointManager # type: ignore

import collections
import json
import pandas as pd  # type: ignore
from typing import Tuple, Optional


class WorkerCheckpointManager(CheckpointManager):
    RecoverResult = collections.namedtuple("RecoverResult", ["full_df", "pruned_df", "processeds"])
    MergedCheckpointResult = collections.namedtuple("MergedCheckpointResult",
           ["num_reference_network", "merged_checkpoint_manager", "dataframe", "task_checkpoint_managers"])

    # Specialization of CheckpointManager for executeTask to checkpoint Task results

    def __init__(self, path:str, is_report:bool=True, is_initialize:bool=False)->None:
        """
        Args:
            subnet_finder (SubnetFinder): SubnetFinder instance
            path (str): Path to the CSV file
            is_report (bool): If True, reports progress
        """
        super().__init__(path, is_report=is_report, is_initialize=is_initialize)

    def recover(self)->RecoverResult:
        """
        Recovers a previously saved DataFrame. The recovered dataframe deletes entries with model strings that are null.

        Returns:
            pd.DataFrame: DataFrame of the checkpoint
            pd.DataFrame: DataFrame of the checkpoint stripped of null entries
            np.ndarray: List of processed tasks
        """
        df = super().recover()
        if len(df) > 0:
            full_df = pd.read_csv(self.path)
            pruned_df, processeds = WorkerCheckpointManager.prune(full_df)
            # Convert the JSON string to a dictionary
            if len(pruned_df) > 0:
                pruned_df.loc[:, cn.FINDER_NAME_DCT] = pruned_df[cn.FINDER_NAME_DCT].apply(lambda x: json.loads(x))
        else:
            full_df = pd.DataFrame()
            pruned_df = pd.DataFrame()
            processeds = []
        self._print(f"Recovering {len(processeds)} processed models from {self.path}")
        return self.RecoverResult(full_df=full_df, pruned_df=pruned_df, processeds=processeds)

    @staticmethod 
    def prune(df:pd.DataFrame)->Tuple[pd.DataFrame, list]:
        """
        Prunes a DataFrame to include only rows where the reference network is not the null string.

        Args:
            df (pd.DataFrame): Table of matching networks
                reference_model (str): Reference model name
                target_model (str): Target model name
                reference_network (str): may be the null string
                target_network (str): may be the null string

        Returns:
            pd.DataFrame: Pruned DataFrame
            list: List of reference networks that were pruned
        """
        is_null = df[cn.FINDER_REFERENCE_NETWORK].isnull()
        is_null_str = df[cn.FINDER_REFERENCE_NETWORK] == cn.NULL_STR
        not_sel = is_null | is_null_str
        reference_networks = list(set(df[not_sel][cn.FINDER_REFERENCE_NETWORK].values))
        return df[~not_sel], reference_networks

    @staticmethod
    def makeTaskPath(outpath_base, task_idx:int)->str:
        """
        Constructs the checkpoint path for a task.

        Args:
            outpath_base (str): Path for the base file
            task_idx (int): Index of the task

        Returns:
            str: Path to the checkpoint
        """
        splits = outpath_base.split(".")
        outpath_pat = splits[0] + "_%d." + splits[1]
        return outpath_pat % task_idx
    
    @classmethod
    def merge(cls, base_checkpoint_path:str, num_task:int,
        merged_checkpoint_result:Optional[MergedCheckpointResult]=None,
        is_report:bool=True)->MergedCheckpointResult:
        """
        Merges the checkpoints from checkpoint managers. Assumes that task checkpoint files are named
        with the pattern base_checkpoint_path_%d.csv.

        Args:
            outpath_base (str): Base path for the checkpoint files
            num_task (int): Number of tasks
            merged_checkpoint_result (Optional[MergedCheckpointResult]): Previous Merged checkpoint result
            is_report (bool): If True, reports progress

        Returns: MergedCheckpointResult
            int: Number of merged entries
            CheckpointManager: Checkpoint manager for merged checkpoint
            pd.DataFrame
        """
        if merged_checkpoint_result is None:
            merged_checkpoint_manager = WorkerCheckpointManager(base_checkpoint_path, is_report=is_report)
            task_checkpoint_managers = [WorkerCheckpointManager(
                    WorkerCheckpointManager.makeTaskPath(base_checkpoint_path, i),
                    is_report=is_report, is_initialize=False)
                    for i in range(num_task)]
        else:
            merged_checkpoint_manager = merged_checkpoint_result.merged_checkpoint_manager
            task_checkpoint_managers = merged_checkpoint_result.task_checkpoint_managers
        full_df = pd.concat([m.recover().full_df for m in task_checkpoint_managers], ignore_index=True)
        merged_checkpoint_manager.checkpoint(full_df)
        #
        if len(full_df) > 0:
            num_reference_network = len(set(full_df[cn.FINDER_REFERENCE_NETWORK].values))
        else:
            num_reference_network = 0
        result = cls.MergedCheckpointResult(
                num_reference_network=num_reference_network,
                merged_checkpoint_manager=merged_checkpoint_manager,
                task_checkpoint_managers=task_checkpoint_managers,
                dataframe=full_df)           
        return result