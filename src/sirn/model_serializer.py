'''Serizalizes Antimony and SBML models as Networks. Can run as a main program.'''

import sirn.constants as cn  # type: ignore
from sirn.network import Network  # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore

import os
import pandas as pd # type: ignore
import argparse
from typing import Optional

SERIALIZATION_FILENAME = "network_serializers.txt"


BATCHSIZE = 20

class ModelSerializer(object):

#    def __init__(self, model_directory:str, model_parent_dir:str=cn.OSCILLATOR_PROJECT,
#                 serialization_path:Optional[str]=None)->None:
#        """
#        Args:
#            model_directory (str): Directory that contains the model files
#            model_directory_parent (Optional[str], optional): Path to the model_directory. If None,
#                the model_directory is a complete path.
#            serialization_file (Optional[str], optional): Path for file where serialization results are stored
#        """
#        if (model_parent_dir is not None) and ("/" in model_directory):
#            raise ValueError("model_directory should not contain a path.")
#        self.model_directory = model_directory
#        self.model_parent_dir = model_parent_dir
#        if serialization_path is None:
#            serialization_path = os.path.join(cn.DATA_DIR, f'{self.model_directory}_serializers.txt')
#        self.serialization_file = serialization_path

    def __init__(self, model_directory:str, serialization_file:str)->None:
        """
        Args:
            model_directory (str): Path to directory that contains the model files
            serialization_file (str): Path for file where serialization results are stored
        """
        self.model_directory = model_directory
        self.serialization_file = serialization_file

    @classmethod
    def makeOscillatorSerializer(cls, oscillator_directory:str)->'ModelSerializer':
        """
        Creates a serializer for the oscillators.

        Args:
            oscillator_directory (str): Name of oscillator directory

        Returns:
            ModelSerializer: _description_
        """
        model_directory = os.path.join(cn.OSCILLATOR_PROJECT, oscillator_directory)
        serialization_file = os.path.join(model_directory, SERIALIZATION_FILENAME)
        return cls(model_directory, serialization_file)

    def serialize(self, batch_size:int=BATCHSIZE, num_batch:Optional[int]=None,
                           report_interval:Optional[int]=10)->None:
        """
        Serializes Antimony models in a directory.

        Args:
            batch_size (int): Number of models to process in a batch
            num_batch (Optional[int]): Number of batches to process
            report_interval (int): Interval to report progress. 
        """
        # Check if there is an existing output file
        processed_network_names:list = []
        if os.path.exists(self.serialization_file):
            with open(self.serialization_file, 'r') as f:
                serialization_strs = f.readlines()
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
            with open(self.serialization_file, 'a') as f:
                for network in network_collection.networks:
                    processed_network_names.append(network.network_name)
                    f.write(f'{network.serialize()}\n')
        print("Done!")

    def deserialize(self)->NetworkCollection:
        """Deserializes the network collection."""
        with open(self.serialization_file, 'r') as f:
            serialization_strs = f.readlines()
        networks = []
        for serialization_str in serialization_strs:
            network = Network.deserialize(serialization_str)
            if network.num_species > 0:
                networks.append(network)
        return NetworkCollection(networks, directory=self.model_directory)

# Run as a main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serialize Antimony Models')
    parser.add_argument('model_directory', type=str, help='Name of directory')
    args = parser.parse_args()
    serializer = ModelSerializer(args.model_directory)
    serializer.serialize(report_interval=BATCHSIZE)