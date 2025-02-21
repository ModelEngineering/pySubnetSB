'''Provides the API for the analysis of structural identity.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.cluster_builder import ClusterBuilder  # type: ignore
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.network import Network, StructuralAnalysisResult  # type: ignore
from pySubnetSB.network_collection import NetworkCollection  # type: ignore
from pySubnetSB.subnet_finder import SubnetFinder  # type: ignore
import pySubnetSB.util as util # type: ignore

import os
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore
import tempfile # type: ignore
from typing import Union, Tuple, Optional

# Ways in which a model can be specified
SPECIFICATION_TYPES = ["antstr", "antfile", "sbmlstr", "sbmlfile", "sbmlurl", "roadrunner"]
DEFAULT_SPECIFICATION_TYPE = "antstr"


#############################
def _getNetworkCollection(directory:str, is_report:bool=True)->Tuple[NetworkCollection, Optional[str]]:
    """Returns the NetworkCollection from the directory.

    Args:
        directory (str): Path to a directory or a serialization file
        is_report (bool): If True, report progress

    Returns:
        NetworkCollection: contains a list of networks
        Optional[str]: Path to the serialization file
    """
    if directory.endswith(".txt"):
        if "http" in directory:
            strings = util.getStringsFromURL(directory)
            networks = ModelSerializer.deserializeFromStrings(strings)
            network_collection = NetworkCollection(networks)
            return network_collection, None
        else:
            serialization_path = directory
        serializer = ModelSerializer(None, serialization_path)
        filename = None
    else:
        # Create the serialization file
        fp = tempfile.NamedTemporaryFile(delete=False)
        filename = fp.name
        fp.close()
        serializer = ModelSerializer(directory, filename)
        if is_report:
            report_interval = 10
        else:
            report_interval = None
        serializer.serialize(report_interval=report_interval)
    return serializer.deserialize(), filename


#############################
class ModelSpecification(object):
    """Specification of a model for a structural analysis."""
    def __init__(self, model:str, specification_type=DEFAULT_SPECIFICATION_TYPE)->None:
        """
        Args:
            model (str): Antimony model or SBML model or path to model
            specification_type:
                "antstr": Antimony model string
                "antfile": Antimony model file
                "sbmlstr": SBML model string
                "sbmlfile": SBML model file
                "sbmlurl": SBML model URL
                "roadrunner": RoadRunner model
        """
        self.model = model
        self.specification_type = specification_type
        if not specification_type in SPECIFICATION_TYPES:
            raise ValueError(f"Invalid model type: {specification_type}. Must be one of {SPECIFICATION_TYPES}")
        
    def __repr__(self)->str:
        return f"ModelReference({self.model}, {self.specification_type})"

    @classmethod 
    def makeNetwork(cls, model:Union[str, 'ModelSpecification'],
          specification_type:Optional[str]=DEFAULT_SPECIFICATION_TYPE)->Network:
        """Returns the Antimony model."""
        if isinstance(model, ModelSpecification):
            specification = model
        else:
            specification = cls(model, specification_type=specification_type)
        network = specification.getNetwork()
        return network
    
    def getNetwork(self)->Network:
        """Returns the Antimony model."""
        if self.specification_type == "antstr":
            antimony_str = self.model
        elif self.specification_type == "antfile":
            with open(self.model, "r") as fd:
                antimony_str = fd.read()
        elif self.specification_type == "roadrunner":
            network = Network.makeFromAntimonyStr(None, roadrunner=self.model)
            return network
        else:
            rr = te.loadSBMLModel(self.model)
            antimony_str = rr.getAntimony()
            antimony_str = antimony_str.replace("\n", ";\n")
        #
        network = Network.makeFromAntimonyStr(antimony_str)
        return network


#############################
def findReferenceInTarget(
      reference_model:Union[str, ModelSpecification],
      target_model:Union[str, ModelSpecification],
      is_subnet:bool=True,
      num_process:int=-1,
      max_num_assignment:int=int(1e12),
      identity:str=cn.ID_STRONG,
      is_report:bool=True)->StructuralAnalysisResult:
    """
    Searches the target model to determine if the reference model is a subnet (if is_subnet is True) or
    if the reference model is structurally identical to the target model (if is_subnet is False).

    Args:
        reference_model (str/ModelSpecification): Antimony str or other model reference
        target_model (str/ModelSpecification): Antimony str or other model reference
        is_subnet (bool): If True, the reference model is a subnet of the target model;
                          otherwise, the reference model is structurally identical to the target model.
        num_process (int): Number of processes to use. If -1, use all available processors.
        max_num_assignment (int): Maximum number of assignment pairs
        identity (str): cn.ID_STRONG (renaming) or cn.ID_WEAK (behavior equivalence)
        is_report (bool): If True, report progress

    Returns:
        StructuralAnalysisResult: List of StructuralAnalysisResult
            assignment_pairs: Assignments of target species/reactions to the reference that result in structural identity
            is_truncated: If True, the search was truncated
            num_species_candidate: Number of assignments of target species to reference species that satisfy constraints
            num_reaction_candidate: Number of assignments of target reaction to reference reaction that satisfy constraints
    """
    reference_network = ModelSpecification.makeNetwork(reference_model)
    target_network = ModelSpecification.makeNetwork(target_model)
    return reference_network.isStructurallyIdentical(
            target_network,
            is_subnet=is_subnet,
            num_process=num_process,
            identity=identity,
            max_num_assignment=max_num_assignment,
            is_report=is_report)

#############################
def clusterStructurallyIdenticalModelsInDirectory(
      model_dir:str,
      identity:str=cn.ID_STRONG,
      max_num_assignment:int=int(1e12),
      is_report:bool=True)->pd.DataFrame:
    """
    Forms clusters of structurally identical models.

    Args:
        model_dir (ModelSpecification/str): Directory with Antimony models or SBML models or a serialization file
        max_num_assignment (int): Maximum number of assignment pairs
        identity (str): cn.ID_STRONG (renaming) or cn.ID_WEAK (behavior equivalence)
        is_report (bool): If True, report progress

    Returns: pd.DataFrame: DataFrame with the subnets. Columns:
        network_name: Name of first network in the cluster collection
        processing_time: Time to process this collection
        is_indeterminate: Indeterminante because too many possible assignments
        assignment_collection: List of networks in the collection
    """
    network_collection, _ = _getNetworkCollection(model_dir, is_report=is_report)
    builder = ClusterBuilder(network_collection,
               identity=identity,
               max_num_assignment=max_num_assignment,
               is_report=is_report)
    builder.cluster()
    # Construct the result
    dct:dict = {k: [] for k in [cn.S_PROCESSING_TIME, cn.S_HASH_VAL, cn.S_PROCESSED_NETWORKS, cn.S_IS_INDETERMINATE]}
    for processed_network_collection in builder.processed_network_collections:
        dct[cn.S_PROCESSING_TIME].append(processed_network_collection.processing_time)
        dct[cn.S_HASH_VAL].append(processed_network_collection.hash_val)
        network_names = [c.network_name for c in processed_network_collection.processed_networks]
        dct[cn.S_PROCESSED_NETWORKS].append(network_names)
        is_indeterminates = [c.is_indeterminate for c in processed_network_collection.processed_networks]
        dct[cn.S_IS_INDETERMINATE].append(any(is_indeterminates))
    return pd.DataFrame(dct)


#############################
def findReferencesInTargets(
      reference_dir:str,
      target_dir:str,
      identity:str=cn.ID_STRONG,
      num_process:int=-1,
      max_num_assignment:int=int(1e12),
      is_report:bool=True)->pd.DataFrame:
    """
    Searches the target directory to determine if the reference directory contains subnets (if is_subnet is True) or
    if the reference directory contains models that are structurally identical to the target directory
    (if is_subnet is False).

    Args:
        reference_dir (str): Directory with Antimony models or SBML models or a serialization file
        target_dir (str): Directory with Antimony models or SBML models or a serialization file
        is_subnet (bool): If True, the reference directory contains subnets of the target directory;
                          otherwise, the reference directory contains models that are structurally identical to the target directory.
        num_process (int): Number of processes to use. If -1, use all available processors.
        max_num_assignment (int): Maximum number of assignment pairs
        identity (str): cn.ID_STRONG (renaming) or cn.ID_WEAK (behavior equivalence)
        is_report (bool): If True, report progress

    Returns:
        pd.DataFrame: DataFrame with the subnets
    """
    # Process the request
    reference_tempfile, target_tempfile = None, None
    try: 
        reference_collection, reference_tempfile = _getNetworkCollection(reference_dir, is_report=is_report)
        target_collection, target_tempfile = _getNetworkCollection(target_dir, is_report=is_report)
        finder = SubnetFinder.makeFromCombinations(
              reference_collection.networks,
              target_collection.networks,
              identity=identity,
              num_process=num_process)
        df = finder.find(is_report=is_report, max_num_assignment=max_num_assignment)
    except:
        df = pd.DataFrame()
    finally:
        if reference_tempfile is not None:
            os.remove(reference_tempfile)
        if target_tempfile is not None:
            os.remove(target_tempfile)
    return df

def makeSerializationFile(directory:str, serialization_path:str, report_interval:int=10,
      is_report:bool=True)->None:
    """
    Serializes the models in the directory to a serialization file.

    Args:
        directory (str): Directory with Antimony models or SBML models
        serialization_path (str): Path to the serialization file to create (or overwrite)
        report_interval (int): Interval to report progress
        is_report (bool): If True, report progress
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a directory")
    if not is_report:
        report_interval = int(1e6)
    #
    serializer = ModelSerializer(directory, serialization_path)
    serializer.serialize(report_interval=report_interval)