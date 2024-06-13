'''Structures the rows and columns of a matrix based on order independent properties of the rows and columns.'''


from sirn.util import hashArray  # type: ignore
from sirn.matrix import Matrix # type: ignore
from sirn.array_collection import ArrayCollection # type: ignore
from sirn import util  # type: ignore

import collections
import numpy as np
import os
import pandas as pd # type: ignore
import tellurium as te   # type: ignore
from typing import List, Optional, Dict

ANTIMONY_EXTS = [".ant", ".txt"]  # Antimony file extensions:95
MODEL_NAME = 'model_name'
ARRAY = 'array'
ROW_NAMES = 'row_names'
COLUMN_NAMES = 'column_names'
SERIALIZATION_NAMES = [MODEL_NAME, ARRAY, ROW_NAMES, COLUMN_NAMES]


class PermutableMatrixSerialization(object):

    def __init__(self,
                 model_name:str,
                 array: np.ndarray,
                 row_names:List[str],
                 column_names:List[str],
                 ):
        self.array = np.array(array.tolist())
        self.row_names = row_names
        self.column_names = column_names
        self.model_name = model_name

    def __repr__(self)->str:
        array_str =  str(self.array)
        array_str = array_str.replace('\n', ',')
        array_str = array_str.replace(' ', ', ')
        while True:
            if ",," not in array_str:
                break
            array_str = array_str.replace(",,", ",")
        row_str = str(self.row_names)
        column_str = str(self.column_names)
        return f'["{self.model_name}", "{array_str}", "{row_str}", "{column_str}"]'

    @classmethod
    def makeDataFrame(cls, ordered_marix_serializations: list):  # type: ignore
        """Constructs a DataFrame from a list of PermutableMatrixSerialization.

        Args:
            ordered_marix_serializations (List[PermutableMatrixSerialization]): _description_

        Returns:
            pd.DataFrame: _description_
        """
        dct: Dict[str, list] = {n: [] for n in SERIALIZATION_NAMES}
        for serialization in ordered_marix_serializations:
            for name in SERIALIZATION_NAMES:
                dct[name].append(getattr(serialization, name))
        return pd.DataFrame(dct)


class PermutableMatrix(Matrix):
        
    def __init__(self, array: np.ndarray,
                 row_names:Optional[List[str]]=None,
                 column_names:Optional[List[str]]=None,
                 model_name:Optional[str]=None): 
        # Inputs
        super().__init__(array)
        if row_names is None:
            row_names = [str(i) for i in range(self.num_row)]  # type: ignore
        if column_names is None:
            column_names = [str(i) for i in range(self.num_column)]  # type: ignore
        if model_name is None:
            model_name = str(np.random.randint(1000000))
        self.row_names = row_names
        self.column_names = column_names
        self.model_name = model_name
        # Outputs
        self.row_collection = ArrayCollection(self.array)
        column_arr = np.transpose(self.array)
        self.column_collection = ArrayCollection(column_arr)
        hash_arr = np.concatenate((self.row_collection.encoding_arr, self.column_collection.encoding_arr))
        self.hash_val = hashArray(hash_arr)

    def __eq__(self, other)->bool:
        """Check if two PermutableMatrix have the same values

        Returns:
            bool: True if the matrix
        """
        if not super().__eq__(other):
            return False
        if not all([s == o] for s, o in zip(self.row_names, other.row_names)):
            return False
        if not all([s == o] for s, o in zip(self.column_names, other.column_names)):
            return False
        if not all([s == o] for s, o in zip(self.row_collection.encoding_arr,
                other.row_collection.encoding_arr)):
            return False
        if not all([s == o] for s, o in zip(self.column_collection.encoding_arr,
                other.column_collection.encoding_arr)):
            return False
        if not self.model_name == other.model_name:
            return False
        return True

    def __repr__(self)->str:
        return str(self.array) + '\n' + str(self.row_collection) + '\n' + str(self.column_collection)
    
    def isPermutablyIdentical(self, other) -> bool:  # type: ignore
        """
        Check if the matrices are permutably identical.
        Order other matrix

        Args:
            other (PermutableMatrix)
        Returns:
            bool
        """
        # Check compatibility
        if not self.isCompatible(other):
            return False
        # The matrices have the same shape and partitions
        #  Order the other matrix to align the partitions of the two matrices
        other_row_itr = other.row_collection.partitionPermutationIterator()
        other_column_itr = other.column_collection.partitionPermutationIterator()
        other_row_perm = next(other_row_itr)
        other_column_perm = next(other_column_itr)
        other_matrix = other.array[other_row_perm, :]
        other_matrix = other_matrix[:, other_column_perm]
        # Search all partition constrained permutations of this matrix to match the other matrix
        row_itr = self.row_collection.partitionPermutationIterator()
        count = 0
        array = self.array.copy()
        for row_perm in row_itr:
            column_itr = self.column_collection.partitionPermutationIterator()
            for col_perm in column_itr:
                count += 1
                matrix = array[row_perm, :]
                matrix = matrix[:, col_perm]
                if np.all(matrix == other_matrix):
                    return True
        return False
    
    def isCompatible(self, other)->bool:
        """
        Checks if the two matrices have the same DimensionClassifications for their rows and columns

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        is_true = self.row_collection.isCompatible(other.row_collection)
        is_true = is_true and self.column_collection.isCompatible(other.column_collection)
        return is_true
    
    def _serializeOne(self)->PermutableMatrixSerialization:
        """Provides a list from which the ordered matrix can be constructed.

        Returns:
            SerializationString
               model_name
               str(self.array)
               row_names
               column_names
        """
        result = PermutableMatrixSerialization(self.model_name, self.array, self.row_names, self.column_names)
        return result

    @classmethod 
    def serializeMany(cls, permutable_matrices:list)->pd.DataFrame:
        """Provides a list from which the ordered matrix can be constructed.

        Returns:
            SerializationString
               model_name
               str(self.array)
               row_names
               column_names
        """
        #serializations = [m.serializeOne() for m in permutable_matrices]
        serializations = []
        for permutable_matrix in permutable_matrices:
            serializations.append(permutable_matrix._serializeOne())
        df = PermutableMatrixSerialization.makeDataFrame(serializations)
        return df
    
    @classmethod
    def deserializeAntimonyFile(cls, path:str):
        """Provides a list from which the ordered matrix can be constructed.

        Args:
            path (str): Path to the antimony model file

        Returns:
            PermutableMatrix
        """
        rr = te.loada(path)
        model_name = os.path.split(path)[1]
        model_name = model_name.split('.')[0]
        #
        row_names = rr.getFloatingSpeciesIds()
        column_names = rr.getReactionIds()
        #
        named_array = rr.getFullStoichiometryMatrix()
        array =  np.array(named_array.tolist())
        #
        permutable_matrix = cls(array, row_names=row_names, column_names=column_names, model_name=model_name)
        return permutable_matrix

    @classmethod
    def deserializeAntimonyDirectory(cls, indir_path:str)->list:
        """Deserializes a directory of antimony models to a DataFrame.

        Args:
            indir_path (str): Path to the antimony model directory

        Returns:
            list-PermutableMatrix
        """
        ffiles = os.listdir(indir_path)
        serialization_strs = []
        for ffile in ffiles:
            if not any([ffile.endswith(ext) for ext in ANTIMONY_EXTS]):
                continue
            ffile = os.path.join(indir_path, ffile)
            serialization_strs.append(cls.deserializeAntimonyFile(ffile))
        df = PermutableMatrixSerialization.makeDataFrame(serialization_strs)
        return df

    @classmethod 
    def constructFromPermutableMatrixSerialization(cls, serialization:PermutableMatrixSerialization):  # type: ignore
        """Constructs an PermutableMatrix from a SerializationString.

        Args:
            SerializationString

        Returns:
            PermutableMatrix
        """
        return cls(serialization.array,
                   row_names=serialization.row_names,
                   column_names=serialization.column_names,
                   model_name=serialization.model_name
                   )
    
    @classmethod 
    def deserializeCSV(cls, path:str)->list:  # type: ignore
        """Constructs a lit of PermutableMatrix from a CSV file.

        Args:
            path (str): Path to the CSV file of SerializationString
        Returns:
            list-PermutableMatrix
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')
        df = pd.read_csv(path)
        return cls.deserializeDataFrame(df)
    
    @classmethod 
    def deserializeDataFrame(cls, df:pd.DataFrame)->list:  # type: ignore
        """Deserializes a DataFrame to a list of PermutableMatrix.

        Args:
            df: pd.DataFrame

        Returns:
            list-PermutableMatrix
        """
        permutable_matrices = []
        for _, row in df.iterrows():
            array = row[ARRAY]
            permutable_matrix = PermutableMatrix(
                array,
                model_name=row[MODEL_NAME],
                row_names=row[ROW_NAMES],
                column_names=row[COLUMN_NAMES],
            )
            permutable_matrices.append(permutable_matrix)
        return permutable_matrices