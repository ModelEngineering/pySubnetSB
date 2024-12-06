'''A container for structurally identical networks and their statistics. Has a string representation.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB import constants as cn  # type: ignore
from pySubnetSB.assignment_pair import AssignmentPair  # type: ignore

import json
import time
from typing import Union, List, Optional


class ProcessedNetwork(object):

    def __init__(self, network_name:Union[Network, str])->None:
        self.network_name = self.convertToNetworkName(network_name)
        # Calculated
        self.is_indeterminate:bool = False
        self.processing_time:float = 0
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

    def addProcessingTime(self, processing_time:float)->None:
        """Adds to processing time

        Args:
            processing_time (Optional[float], optional): _description_. Defaults to None.
        """
        self.processing_time += processing_time

    def __eq__(self, other:object)->bool:
        if not isinstance(other, ProcessedNetwork):
            return False
        return (self.network_name == other.network_name and
                self.is_indeterminate == other.is_indeterminate)
    
    def copy(self)->'ProcessedNetwork':
        processed_network = ProcessedNetwork(self.network_name)
        processed_network.processing_time = self.processing_time
        processed_network.is_indeterminate = self.is_indeterminate
        processed_network.assignment_collection = [a.copy() for a in self.assignment_collection]
        return processed_network
    
#    def __repr__(self)->str:
#        repr_str = self._CSV_MAKER.encode(
#            network_name=self.network_name,
#            processing_time=self.processing_time,
#            is_indeterminate=self.is_indeterminate,
#            assignment_collection=self.assignment_collection)
#        return repr_str

#    @classmethod
#    def makeFromRepr(cls, repr_str:str)->'processedNetwork':
#        """
#        Constructs a processedNetwork from a string representation.
#
#        Args:
#            repr_str (str): _description_
#
#        Returns:
#            processedNetwork
#        """
#        dct = cls._CSV_MAKER.decode(repr_str)
#        processed_network = processedNetwork(dct[NETWORK_NAME])
#        processed_network.setAssignmentCollection(dct[ASSIGNMENT_COLLECTION])
#        processed_network.setProcessingTime(dct[PROCESSING_TIME])
#        processed_network.setIndeterminate(dct[IS_INDETERMINATE])
#        return processed_network
    
    def serialize(self)->str:
        """Creates a JSON string for the object.

        Returns:
            str
        """
        assignment_collection = [a.serialize() for a in self.assignment_collection]
        dct = {cn.S_ID: self.__class__.__name__,
               cn.S_NETWORK_NAME: self.network_name,
               cn.S_PROCESSING_TIME: self.processing_time,
               cn.S_IS_INDETERMINATE: self.is_indeterminate,
               cn.S_ASSIGNMENT_COLLECTION: assignment_collection}
        return json.dumps(dct)
    
    @classmethod
    def deserialize(cls, serialization_str)->'ProcessedNetwork':
        """Creates a processed network from a JSON serialization string.

        Args:
            serialization_str

        Returns:
            processedNetwork
        """
        dct = json.loads(serialization_str)
        if not cls.__name__ in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        processed_network = cls(dct[cn.S_NETWORK_NAME])
        processed_network.addProcessingTime(dct[cn.S_PROCESSING_TIME])
        processed_network.setIndeterminate(dct[cn.S_IS_INDETERMINATE])
        processed_network.setAssignmentCollection(dct[cn.S_ASSIGNMENT_COLLECTION])
        return processed_network