'''A container for structurally identical networks and their statistics. Has a string representation.'''

import sirn.constants as cn # type: ignore
from sirn.network import Network  # type: ignore
from sirn import constants as cn  # type: ignore
from sirn.archive2.csv_maker import CSVMaker  # type: ignore
from sirn.assignment_pair import AssignmentPair  # type: ignore

import json
import time
from typing import Union, List, Optional


class ClusteredNetwork(object):

    def __init__(self, network_name:Union[Network, str], processing_time:Optional[float]=None)->None:
        self.network_name = self.convertToNetworkName(network_name)
        # Calculated
        if processing_time is None:
            self.processing_time = time.process_time()  # Start time for processing the network
        else:
            self.processing_time = processing_time  # Specify the full processing time
        self.is_indeterminate:bool = False
        self.assignment_collection:List[AssignmentPair] = []

    def __repr__(self)->str:
        return f"{self.network_name}_{self.processing_time}_{self.is_indeterminate}_{self.assignment_collection}"

    @staticmethod 
    def convertToNetworkName(network:Union[Network, str])->str:
        """Returns the network name

        Args:
            network (Union[Network, str]): _description_

        Returns:
            str: _description_
        """
        if isinstance(network, str):
            network_name = network
        else:
            network_name = network.network_name
        return network_name

    def setIndeterminate(self, value)->None:
        self.is_indeterminate = value

    def setAssignmentCollection(self, assignment_collection:List[AssignmentPair])->None:
        self.assignment_collection = assignment_collection

    def setProcessingTime(self, processing_time:Optional[float]=None)->None:
        if processing_time is None:
            self.processing_time = time.process_time() - self.processing_time
            if self.processing_time < 0:
                raise ValueError(f"Processing time is negative: {self.processing_time}")
        else:
            self.processing_time = processing_time

    def __eq__(self, other:object)->bool:
        if not isinstance(other, ClusteredNetwork):
            return False
        return (self.network_name == other.network_name and
                self.is_indeterminate == other.is_indeterminate)
    
    def copy(self)->'ClusteredNetwork':
        clustered_network = ClusteredNetwork(self.network_name)
        clustered_network.processing_time = self.processing_time
        clustered_network.is_indeterminate = self.is_indeterminate
        clustered_network.assignment_collection = [a.copy() for a in self.assignment_collection]
        return clustered_network
    
#    def __repr__(self)->str:
#        repr_str = self._CSV_MAKER.encode(
#            network_name=self.network_name,
#            processing_time=self.processing_time,
#            is_indeterminate=self.is_indeterminate,
#            assignment_collection=self.assignment_collection)
#        return repr_str

#    @classmethod
#    def makeFromRepr(cls, repr_str:str)->'ClusteredNetwork':
#        """
#        Constructs a ClusteredNetwork from a string representation.
#
#        Args:
#            repr_str (str): _description_
#
#        Returns:
#            ClusteredNetwork
#        """
#        dct = cls._CSV_MAKER.decode(repr_str)
#        clustered_network = ClusteredNetwork(dct[NETWORK_NAME])
#        clustered_network.setAssignmentCollection(dct[ASSIGNMENT_COLLECTION])
#        clustered_network.setProcessingTime(dct[PROCESSING_TIME])
#        clustered_network.setIndeterminate(dct[IS_INDETERMINATE])
#        return clustered_network
    
    def serialize(self)->str:
        """Creates a JSON string for the object.

        Returns:
            str
        """
        assignment_collection = [a.serialize() for a in self.assignment_collection]
        dct = {cn.S_ID: str(self.__class__),
               cn.S_NETWORK_NAME: self.network_name,
               cn.S_PROCESSING_TIME: self.processing_time,
               cn.S_IS_INDETERMINATE: self.is_indeterminate,
               cn.S_ASSIGNMENT_COLLECTION: assignment_collection}
        return json.dumps(dct)
    
    @classmethod
    def deserialize(cls, serialization_str)->'ClusteredNetwork':
        """Creates a clustered network from a JSON serialization string.

        Args:
            serialization_str

        Returns:
            ClusteredNetwork
        """
        dct = json.loads(serialization_str)
        if not str(cls) in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        clustered_network = cls(dct[cn.S_NETWORK_NAME])
        clustered_network.setProcessingTime(dct[cn.S_PROCESSING_TIME])
        clustered_network.setIndeterminate(dct[cn.S_IS_INDETERMINATE])
        clustered_network.setAssignmentCollection(dct[cn.S_ASSIGNMENT_COLLECTION])
        return clustered_network