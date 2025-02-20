'''Serizalizes Antimony and SBML models as Networks. Can run as a main program.'''

import pySubnetSB.constants as cn  # type: ignore
from pySubnetSB.network import Network  # type: ignore
from pySubnetSB.network_collection import NetworkCollection  # type: ignore

import os
import pandas as pd # type: ignore
import argparse
from typing import Optional, List

DEFAULT_SERIALIZATION_FILENAME = "network_serializers.txt"


BATCHSIZE = 20

class ModelSerializer(object):

    def __init__(self, model_directory:Optional[str]=None,
          serialization_path:str=DEFAULT_SERIALIZATION_FILENAME)->None:
        """
        The constructor has two forms:
        1. A model directory is given and optionally a name is given for the serialization file.
        2. A serialization path is given and the model directory is None.
        Args:
            model_directory (str): Path to directory that contains the model files
            serialization_file (str): Path for file where serialization results are stored
        """
        self.model_directory = model_directory
        self.serialization_file = serialization_path
        # self.serialization_path is the path to the serialization file
        if model_directory is None:
            if serialization_path.endswith("txt"):
                self.serialization_path = serialization_path
            else:
                raise ValueError("model_directory should be set if serialization_file is not a txt file.")
        else:
            if "/" in serialization_path:
                self.serialization_path = serialization_path
            else:
                self.serialization_path = os.path.join(model_directory, serialization_path)

    def remove(self)->None:
        """Removes the serialization file."""
        if os.path.exists(self.serialization_path):
            os.remove(self.serialization_path)

    @classmethod
    def serializerFromNetworks(cls, networks:List[Network], serialization_file:str,
          is_initialize:bool=False)->None:
        """
        Creates a serializer from a list of networks.

        Args:
            networks (List[Network]): List of networks
            serialization_file (str): Path to serialization file
            is_initialize (bool): If True, initializes the serialization file
        """
        serializer = cls(None, serialization_file)
        serializer.serializeNetworks(networks, is_initialize=is_initialize)

    @classmethod
    def makeOscillatorSerializer(cls, oscillator_directory:str,
          parent_directory:str=cn.OSCILLATOR_PROJECT)->'ModelSerializer':
        """
        Creates a serializer for the oscillators.

        Args:
            oscillator_directory (str): Name of oscillator directory
            parent_directory (str): 

        Returns:
            ModelSerializer: _description_
        """
        model_directory = os.path.join(parent_directory, oscillator_directory)
        return cls(model_directory, DEFAULT_SERIALIZATION_FILENAME)

    def serialize(self, batch_size:int=BATCHSIZE, num_batch:Optional[int]=None,
                           report_interval:Optional[int]=10,
                           serialization_strs:Optional[List[str]]=None)->None:
        """
        Serializes Antimony models in a directory.

        Args:
            batch_size (int): Number of models to process in a batch
            num_batch (Optional[int]): Number of batches to process
            report_interval (int): Interval to report progress. 
        """
        # Check if there is an existing output file
        processed_network_names:list = []
        if serialization_strs is None:
            if os.path.exists(self.serialization_path):
                with open(self.serialization_path, 'r') as f:
                    serialization_strs = f.readlines()
            else:
                serialization_strs = []
        processed_network_names = []
        for serialization_str in serialization_strs:
            network = Network.deserialize(serialization_str)
            processed_network_names.append(network.network_name)
        batch_count = 0
        while True:
            batch_count += 1
            if num_batch is not None and batch_count > num_batch:
                break
            #
            network_collection = NetworkCollection.makeFromAntimonyDirectory(self.model_directory,
                batch_size=batch_size,
                processed_network_names=processed_network_names, report_interval=report_interval)
            if len(network_collection) == 0:    
                break
            with open(self.serialization_path, 'a') as f:
                for network in network_collection.networks:
                    processed_network_names.append(network.network_name)
                    try:
                        stg = f'{network.serialize()}\n'
                        f.write(stg)
                    except:
                        print(f"**Error serializing {network.network_name}")
        if report_interval is not None:
            print("Done!")
    
    def serializeNetworks(self, networks:List[Network], is_initialize:bool=False)->None:
        """
        Serializes Networks
        """
        if is_initialize:
            mode = 'w'
        else:
            mode = 'a'
        with open(self.serialization_path, mode) as f:
            for network in networks:
                f.write(f'{network.serialize()}\n')

    @classmethod
    def deserializeFromStrings(cls, strings:List[str],
          model_names:Optional[List[str]]=None)->List[Network]:
        """Deserializes the network collection from a list of strings.
        Args:
            List of strings
            model_names (List[str]): If set, only networks with these names are deserialized.
        Returns:
            List[Network]: List of networks
        """
        networks = []
        for serialization_str in strings:
            if len(serialization_str) == 0:
                continue
            network = Network.deserialize(serialization_str)
            if network.num_species > 0:
                if (model_names is None) or (network.network_name in model_names):
                    networks.append(network)
        return networks

    def deserialize(self, model_names:Optional[List[str]]=None)->NetworkCollection:
        """Deserializes the network collection.
        Args:
            model_names (List[str]): If set, only networks with these names are deserialized.
        Returns:
            NetworkCollection: Collection of networks
        """
        with open(self.serialization_path, 'r') as f:
            serialization_strs = f.readlines()
        networks = self.deserializeFromStrings(serialization_strs, model_names=model_names)
        return NetworkCollection(networks, directory=self.model_directory)

# Run as a main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serialize Antimony Models')
    parser.add_argument('model_directory', type=str, help='Name of directory')
    args = parser.parse_args()
    serializer = ModelSerializer(args.model_directory)
    serializer.serialize(report_interval=BATCHSIZE)