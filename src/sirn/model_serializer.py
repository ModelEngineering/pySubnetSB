'''Serizalizes Antimony and SBML models as Networks. Can run as a main program.'''

import sirn.constants as cn  # type: ignore
from sirn.network import Network  # type: ignore
from sirn.network_collection import NetworkCollection  # type: ignore

import os
import pandas as pd # type: ignore
import argparse
from typing import Optional


BATCHSIZE = 500

class ModelSerializer(object):

    def __init__(self, model_directory:str, model_parent_dir:str=cn.OSCILLATOR_PROJECT,
                 serialization_file:Optional[str]=None)->None:
        """
        Args:
            model_directory (str): Directory that contains the model files
            model_directory_parent (Optional[str], optional): Path to the model_directory
            serialization_file (Optional[str], optional): Where serialization results are stored
        """
        self.model_directory = model_directory
        self.model_parent_dir = model_parent_dir
        if serialization_file is None:
            serialization_file = os.path.join(cn.DATA_DIR, f'{self.model_directory}_serializers.txt')
        self.serialization_file = serialization_file

    def serialize(self, batch_size:int=BATCHSIZE, num_batch:Optional[int]=None,
                           report_interval:Optional[int]=None)->None:
        """
        Serializes Antimony models in a directory.

        Args:
            batch_size (int): Number of models to process in a batch
            num_batch (Optional[int]): Number of batches to process
            report_interval (int): Interval to report progress. 
        """
        directory_path = os.path.join(self.model_parent_dir, self.model_directory)
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
            network_collection = NetworkCollection.makeFromAntimonyDirectory(directory_path,
                batch_size=batch_size,
                processed_network_names=processed_network_names, report_interval=report_interval)
            if len(network_collection) == 0:    
                break
            with open(self.serialization_file, 'a') as f:
                for network in network_collection.networks:
                    processed_network_names.append(network.network_name)
                    f.write(f'{network.serialize()}\n')
        print("Done!")

    def deserialize(self, serialization_file:Optional[str]=None)->NetworkCollection:
        """Deserializes the network collection."""
        with open(self.serialization_file, 'r') as f:
            serialization_strs = f.readlines()
        networks = []
        for serialization_str in serialization_strs:
            network = Network.deserialize(serialization_str)
            networks.append(network)
        return NetworkCollection(networks, directory=self.model_directory)

# Run as a main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serialize Antimony Models')
    parser.add_argument('model_directory', type=str, help='Name of directory')
    args = parser.parse_args()
    if not args.model_directory in cn.OSCILLATOR_DIRS:
        raise ValueError(f"{args.self.model_directory} not in {cn.OSCILLATOR_DIRS}")
    serializer = ModelSerializer(args.model_directory)
    serializer.serialize(report_interval=BATCHSIZE)