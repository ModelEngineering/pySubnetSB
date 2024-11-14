'''Worker for running SubnetFinder in parallel.'''

import sirn.constants as cn # type: ignore
from sirn.subnet_finder import SubnetFinder # type: ignore
from sirn.model_serializer import ModelSerializer # type: ignore
from sirn.network import Network  # type: ignore
from sirn.checkpoint_manager import CheckpointManager # type: ignore

import collections
import json
import os
import multiprocessing as mp
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Tuple, Optional

MergedCheckpointResult = collections.namedtuple("MergedCheckpointResult",
      ["num_reference_network", "merged_checkpoint_manager", "dataframe", "task_checkpoint_managers"])


def executeTask(task_idx:int, queue, total_task:int, outpath_base:str,
        reference_serialization_path:str, target_serialization_path:str,
        identity:str=cn.ID_STRONG, is_report:bool=True,
        is_initialize:bool=True)->pd.DataFrame:
    """
    Processes pairs of reference and target networks. The reference networks are processed
    based on the index provided in the queue. A task has its own checkpoint file.

    Args:
        task_idx (int): Index of the task
        queue (Queue): Queue of Workunit
        total_task (int): Total number of tasks (including this task)
        outpath_base (str): Output path for the consoldated checkpoint files
        reference_serialization_path (str): Path to the serialized reference networks
        target_serialization_path (str): Path to the serialized target networks
        identity (str): Identity type
        is_report (float): Task 0 reports progress
        is_initialize (bool): If True, initialize the checkpoint

    Returns:
        pd.DataFrame: Table of matching models
            reference_model (str): Reference model name
            target_model (str): Target model name
            reference_network (str): may be the null string
            target_network (str): may be the null string
    """
    #####
    def getNetworks(serialization_path:str)->List[Network]:
        """
        Obtains the networks from a serialization file.

        Args:
            serialization_path (str): Path to the serialization file

        Returns:
            List[Network]: List of networks
        """
        serializer = ModelSerializer(None, serialization_path)
        collection = serializer.deserialize()
        return collection.networks
    #####
    # Calculate the number of processes per task
    num_process = int(mp.cpu_count() / total_task)
    # Get the reference and target models
    reference_networks = getNetworks(reference_serialization_path)
    target_networks = getNetworks(target_serialization_path)
    # Set up the checkpoint manager for this task
    is_report_task = is_report if task_idx == 0 else False
    outpath_task = _CheckpointManager.makeTaskPath(outpath_base, task_idx)
    task_checkpoint_manager = _CheckpointManager(outpath_task, is_report=is_report_task,
          is_initialize=is_initialize)
    # Get the processed reference networks
    full_task_df, _, _ = task_checkpoint_manager.recover()
    # Process the unprocessed reference models in batches
    msg = f"**Task start {task_idx} with"
    msg += " {len(reference_networks) - len(processed_reference_networks)} networks."
    if is_report_task:
        print(msg)
    prior_merged_checkpoint_result = None
    while True:
        workunit = queue.get()
        if workunit.is_done:
            queue.task_done()
            break
        # Extract the networks for this batch
        if workunit.reference_idx < 0:
            batch_reference_networks = reference_networks
        else:
            batch_reference_networks = [reference_networks[workunit.reference_idx]]
        if workunit.target_idx < 0:
            batch_target_networks = target_networks
        else:
            batch_target_networks = [target_networks[workunit.target_idx]]
        # Construct the notification
        if len(batch_reference_networks) == 1:
            msg = f"**Task {task_idx} processing reference network {batch_reference_networks[0].network_name}."
        elif len(batch_target_networks) == 1:
            msg = f"**Task {task_idx} processing target network {batch_target_networks[0].network_name}."
        else:
            msg = f"**Task {task_idx} processing {len(batch_reference_networks)} reference networks."
        if is_report_task:
            print(msg)
        # Finding the Subnets
        finder = SubnetFinder(batch_reference_networks, batch_target_networks, identity=identity,
                num_process=num_process)
        incremental_df = finder.find(is_report=is_report_task)
        full_task_df = pd.concat([full_task_df, incremental_df], ignore_index=True)
        task_checkpoint_manager.checkpoint(full_task_df)
        # Update the processed networks
        merged_checkpoint_result = _mergeCheckpoints(outpath_base, total_task,
              merged_checkpoint_result=prior_merged_checkpoint_result, is_report=is_report_task)
        prior_merged_checkpoint_result = merged_checkpoint_result
        if is_report_task:
            print(f"**Processed {merged_checkpoint_result.num_reference_network} work units.")
    #
    if is_report_task:
        print(f"**Task {task_idx} done.")
    return full_task_df


############################### INTERNAL FUNCTIONS ###############################
def _mergeCheckpoints(outpath_base:str, num_task:int,
      merged_checkpoint_result:Optional[MergedCheckpointResult]=None,
      is_report:bool=True)->MergedCheckpointResult:
    """
    Merges the checkpoints from the checkpoint managers.

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
        merged_checkpoint_manager = CheckpointManager(outpath_base, is_report=is_report)
        task_checkpoint_managers = [CheckpointManager(
                _CheckpointManager.makeTaskPath(outpath_base, i),
                is_report=is_report, is_initialize=False)
                for i in range(num_task)]
    else:
        merged_checkpoint_manager = merged_checkpoint_result.merged_checkpoint_manager
        task_checkpoint_managers = merged_checkpoint_result.task_checkpoint_managers
    full_df = pd.concat([m.recover() for m in task_checkpoint_managers], ignore_index=True)
    merged_checkpoint_manager.checkpoint(full_df)
    #
    if len(full_df) > 0:
        num_reference_network = len(set(full_df[cn.FINDER_REFERENCE_NETWORK].values))
    else:
        num_reference_network = 0
    result = MergedCheckpointResult(
            num_reference_network=num_reference_network,
            merged_checkpoint_manager=merged_checkpoint_manager,
            task_checkpoint_managers=task_checkpoint_managers,
            dataframe=full_df)           
    return result


##############################################################
class _CheckpointManager(CheckpointManager):
    # Specialization of CheckpointManager for executeTask

    def __init__(self, path:str, is_report:bool=True, is_initialize:bool=False)->None:
        """
        Args:
            subnet_finder (SubnetFinder): SubnetFinder instance
            path (str): Path to the CSV file
            is_report (bool): If True, reports progress
        """
        super().__init__(path, is_report=is_report, is_initialize=is_initialize)

    def recover(self)->Tuple[pd.DataFrame, pd.DataFrame, list]:
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
            pruned_df, processed_list = _CheckpointManager.prune(full_df)
            # Convert the JSON string to a dictionary
            if len(pruned_df) > 0:
                pruned_df.loc[:, cn.FINDER_NAME_DCT] = pruned_df[cn.FINDER_NAME_DCT].apply(lambda x: json.loads(x))
        else:
            full_df = pd.DataFrame()
            pruned_df = pd.DataFrame()
            processed_list = []
        self._print(f"Recovering {len(processed_list)} processed models from {self.path}")
        return full_df, pruned_df, processed_list

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
    
#############################################################
class Workunit(object):
    def __init__(self, reference_idx:int=-1, target_idx:int=-1, is_done:bool=False):
        """
        Args:
            reference_idx (int, optional): index of reference network. Defaults to -1 means all.
            target_idx (int, optional): index of target network. Defaults to -1 means all.
            is_done (bool, optional): No more work.
        """
        self.reference_idx = reference_idx
        self.target_idx = target_idx
        self.is_done = is_done

    def __repr__(self):
        return f"Workunit(ref={self.reference_idx}, tgt={self.target_idx}, is_done={self.is_done})"
    
    @classmethod
    def addMultipleWorkunits(cls, queue, reference_idxs:Optional[List[int]]=None,
          target_idxs:Optional[List[int]]=None)->None:
        """
        Adds a list of integer workunits to the queue that is the cross product
        of the reference and target indexes.

        Args:
            queue (Queue): Queue to add the workunits
            reference_idxs (Optional[List[int]]): List of reference indexes
            target_idxs (Optional[List[int]]): List of target indexes
        """
        if reference_idxs is None:
            reference_idxs = [-1]
        if target_idxs is None:
            target_idxs = [-1]
        workunits = [cls(reference_idx=ref_idx, target_idx=tgt_idx)
              for ref_idx in reference_idxs for tgt_idx in target_idxs]
        [queue.put(w) for w in workunits]