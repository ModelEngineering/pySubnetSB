'''Serializes and deserializes PMatrix objects.'''
"""
Serialization is from a PMAtrixCollection to a DataFrame.
Deserialization is from a DataFrame to a PMatrixCollection.
"""

from sirn.pmatrix import PMatrix
from sirn.pmatrix_collection import PMatrixCollection  # type: ignore

import collections
import numpy as np
import os
import pandas as pd  # type: ignore
import tellurium as te # type: ignore
from typing import Dict, Optional, List

ANTIMONY_EXTS = [".ant", ".txt", ""]  # Antimony file extensions:95
MODEL_NAME = 'model_name'
ARRAY_STR = 'array_str'
NUM_ROW = 'num_row'
NUM_COL = 'num_col'
ROW_NAMES = 'row_names'
COLUMN_NAMES = 'column_names'
SERIALIZATION_NAMES = [MODEL_NAME, ARRAY_STR, ROW_NAMES, COLUMN_NAMES, NUM_ROW, NUM_COL, NUM_ROW, NUM_COL]

ArrayContext = collections.namedtuple('ArrayContext', "string, nrow, ncol")


class PMCSerializer(object):

    def __init__(self, pmatrix_collection:PMatrixCollection):
        self.collection = pmatrix_collection

    def __repr__(self)->str:
        names = [p.model_name for p in self.collection.pmatrices]
        return "---".join(names)

    @staticmethod 
    def _array2Context(array:np.ndarray)->ArrayContext:
        nrow, ncol = np.shape(array)
        flat_array = np.reshape(array, nrow*ncol)
        str_arr = [str(i) for i in flat_array]
        array_str = "[" + ",".join(str_arr) + "]"
        return ArrayContext(array_str, nrow, ncol)
    
    @staticmethod
    def _string2Array(array_context:ArrayContext)->np.ndarray:
        array = np.array(eval(array_context.string))
        array = np.reshape(array, (array_context.nrow, array_context.ncol))
        return array

    def serialize(self)->pd.DataFrame:
        """Constructs a MutableCollection to a DataFrame.

        Returns:
            pd.DataFrame: See SERIALIZATION_NAMES
        """
        dct: Dict[str, list] = {n: [] for n in SERIALIZATION_NAMES}
        for pmatrix in self.collection.pmatrices:
            dct[MODEL_NAME].append(pmatrix.model_name)
            array_context = self._array2Context(pmatrix.array)
            dct[ARRAY_STR].append(array_context.string)
            dct[NUM_ROW].append(array_context.nrow)
            dct[NUM_COL].append(array_context.ncol)
            dct[ROW_NAMES].append(str(pmatrix.row_names))
            dct[COLUMN_NAMES].append(str(pmatrix.column_names))
        return pd.DataFrame(dct)
    
    @classmethod 
    def deserialize(cls, df:pd.DataFrame)->PMatrixCollection:  # type: ignore
        """Deserializes a DataFrame to a MutableCollection.

        Args:
            df: pd.DataFrame

        Returns:
            PMatrixCollection
        """
        pmatrices = []
        for _, row in df.iterrows():
            array_str = row[ARRAY_STR]
            num_row = row[NUM_ROW]
            num_col = row[NUM_COL]
            array_context = ArrayContext(array_str, num_row, num_col)
            array = cls._string2Array(array_context)
            pmatrix = PMatrix(
                array,
                model_name=row[MODEL_NAME],
                row_names=row[ROW_NAMES],
                column_names=row[COLUMN_NAMES],
            )
            pmatrices.append(pmatrix)
        return PMatrixCollection(pmatrices)
    
    @classmethod
    def _makePMatrixAntimonyFile(cls, path:str)->PMatrix:
        """Creates a pmatrix from a model in an Antimony file.

        Args:
            path (str): Path to the antimony model file

        Returns:
            pmatrixMatrix
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
        pmatrix_matrix = PMatrix(array, row_names=row_names, column_names=column_names, model_name=model_name)
        return pmatrix_matrix

    @classmethod
    def makePMCollectionAntimonyDirectory(cls, indir_path:str, max_file:Optional[int]=None,
                processed_model_names:Optional[List[str]]=None,
                report_interval:Optional[int]=None)->PMatrixCollection:
        """Creates a pmatrixCollection from a directory of Antimony files.

        Args:
            indir_path (str): Path to the antimony model directory
            max_file (int): Maximum number of files to process
            processed_model_names (List[str]): Names of models already processed
            report_interval (int): Report interval

        Returns:
            pmatrixCollection
        """
        ffiles = os.listdir(indir_path)
        pmatrices = []
        model_names = []
        if processed_model_names is not None:
            model_names = list(processed_model_names)
        for count, ffile in enumerate(ffiles):
            if report_interval is not None and count % report_interval == 0:
                is_report = True
            else:
                is_report = False
            model_name = ffile.split('.')[0]
            if model_name in model_names:
                if is_report:
                    print(".")
                continue
            if (max_file is not None) and (count >= max_file):
                break
            if not any([ffile.endswith(ext) for ext in ANTIMONY_EXTS]):
                continue
            ffile = os.path.join(indir_path, ffile)
            pmatrices.append(cls._makePMatrixAntimonyFile(ffile))
            if is_report:
                print(f"Processed {count} files.")
        return PMatrixCollection(pmatrices)