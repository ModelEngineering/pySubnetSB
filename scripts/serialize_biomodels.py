'''Constructs serialization of BioModels.'''
from pySubnetSB.model_serializer import ModelSerializer  # type: ignore
from pySubnetSB.network import Network  # type: ignore
import pySubnetSB.constants as cn # type: ignore

import os 


BIOMODEL_DIR = "/Users/jlheller/home/Technical/repos/SBMLModel/data"
BIOMODEL_FIILES = [f for f in os.listdir(BIOMODEL_DIR) if f.endswith(".xml")]
SERIALIZATION_FILE = os.path.join(cn.DATA_DIR, "biomodels_serialized.txt")

serializer = ModelSerializer(BIOMODEL_DIR, serialization_path=SERIALIZATION_FILE)
network_collection = serializer.serialize()