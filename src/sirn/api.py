'''Describes the API for the SIRN service.'''

from sirn.subnet_finder import SubnetFinder  # type: ignore
from sirn.model_serializer import ModelSerializer  # type: ignore
from sirn.network import Network, StructuralAnalysisResult  # type: ignore

import os
import pandas as pd  # type: ignore
from typing import List


def findForModels(reference_model:str, target_model:str, identity:str, is_subset:bool=True, num_process:int=-1,
      is_report:bool=True)->List[StructuralAnalysisResult]:
    """
    Searches the target model to determine if the reference model is a subnet (if is_subset is True) or
    if the reference model is structurally identical to the target model (if is_subset is False).

    Args:
        reference_model (str): Antimony model or SBML model or path to model
        target_model (str): Antimony model or SBML model or path to model
        identity (str): cn.ID_STRONG (renaming) or cn.ID_WEAK (behavior equivalence)
        is_subset (bool): If True, the reference model is a subnet of the target model;
                          otherwise, the reference model is structurally identical to the target model.
        num_process (int): Number of processes to use. If -1, use all available processors.
        is_report (bool): If True, report progress

    Returns:
        List[StructuralAnalysisResult]: List of StructuralAnalysisResult
            assignment_pairs: Assignments of target species/reactions to the reference that result in structural identity
            is_truncated: If True, the search was truncated
            num_species_candidate: Number of assignments of target species to reference species that satisfy constraints
            num_reaction_candidate: Number of assignments of target reaction to reference reaction that satisfy constraints
    """
    raise NotImplementedError("findForModels")


def findForDirectories(reference_dir:str, target_dir:str, identity:str, is_subset:bool=True, num_process:int=-1,
      is_report:bool=True)->pd.DataFrame:
    """
    Searches the target directory to determine if the reference directory contains subnets (if is_subset is True) or
    if the reference directory contains models that are structurally identical to the target directory
    (if is_subset is False).

    Args:
        reference_dir (str): Directory with Antimony models or SBML models or a serialization file
        target_dir (str): Directory with Antimony models or SBML models or a serialization file
        identity (str): cn.ID_STRONG (renaming) or cn.ID_WEAK (behavior equivalence)
        is_subset (bool): If True, the reference directory contains subnets of the target directory;
                          otherwise, the reference directory contains models that are structurally identical to the target directory.
        num_process (int): Number of processes to use. If -1, use all available processors.
        is_report (bool): If True, report progress

    Returns:
        pd.DataFrame: DataFrame with the subnets
    """
    raise NotImplementedError("findForDirectories")

def serialize(directory:str, serialization_path:str, report_interval:int=10)->None:
    """
    Serializes the models in the directory to a serialization file.

    Args:
        directory (str): Directory with Antimony models or SBML models
        serialization_path (str): Path to the serialization file to create (or overwrite)
        report_interval (int): Interval to report progress
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a directory")
    #
    serializer = ModelSerializer(directory, serialization_path)
    serializer.serialize(report_interval=report_interval)