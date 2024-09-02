'''A container for structurally identical networks and their statistics. Has a string representation.'''

import sirn.constants as cn # type: ignore
from sirn.network import Network  # type: ignore
from sirn import constants as cn  # type: ignore
from sirn.csv_maker import CSVMaker  # type: ignore
from sirn.assignment_pair import AssignmentCollection  # type: ignore

import collections
import time
from typing import Union, List, Optional
NETWORK_NAME = 'network_name'
PROCESSING_TIME = 'processing_time'
IS_INDETERMINATE = 'is_indeterminate'
ASSIGNMENT_COLLECTION = 'assignment_collection'
CSV_DCT = {NETWORK_NAME: str, PROCESSING_TIME: float,
           IS_INDETERMINATE: bool, ASSIGNMENT_COLLECTION: AssignmentCollection}


class ClusteredNetwork(object):
    _CSV_MAKER = CSVMaker(CSV_DCT)

    def __init__(self, network_name:Union[Network, str], processing_time:Optional[float]=None)->None:
        self.network_name = self.convertToNetworkName(network_name)
        # Calculated
        if processing_time is None:
            self.processing_time = self._getInitialTime()
        else:
            self.processing_time = processing_time
        self.is_indeterminate:bool = False
        self.assignment_collection:AssignmentCollection = AssignmentCollection([])

    def _getInitialTime(self)->float:
        return time.process_time()

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

    def setAssignmentCollection(self, assignment_collection:AssignmentCollection)->None:
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
        clustered_network.assignment_collection = self.assignment_collection
        return clustered_network
    
    def __repr__(self)->str:
        repr_str = self._CSV_MAKER.encode(
            network_name=self.network_name,
            processing_time=self.processing_time,
            is_indeterminate=self.is_indeterminate,
            assignment_collection=self.assignment_collection)
        return repr_str

    @classmethod
    def makeFromRepr(cls, repr_str:str)->'ClusteredNetwork':
        """
        Constructs a ClusteredNetwork from a string representation.

        Args:
            repr_str (str): _description_

        Returns:
            ClusteredNetwork
        """
        dct = cls._CSV_MAKER.decode(repr_str)
        clustered_network = ClusteredNetwork(dct[NETWORK_NAME])
        clustered_network.setAssignmentCollection(dct[ASSIGNMENT_COLLECTION])
        clustered_network.setProcessingTime(dct[PROCESSING_TIME])
        clustered_network.setIndeterminate(dct[IS_INDETERMINATE])
        return clustered_network