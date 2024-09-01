'''Creates a string representation for scalars and arrays.'''

"""
Creates a CSV format for a single CSV line. Can also write the header. Comma separated strings are called
items.

Usage:
    repr = StringRepresentation([("name1", str), ("name2", int)])
    print(repr.makeHeader())
    encode = repr.encode(name1="hello", name2=3)
    value_dct = repr.decode(encode)



"""
import sirn.constants as cn  # type: ignore
from sirn.assignment_pair import AssignmentPair, AssignmentCollection  # type: ignore

import numpy as np
from typing import Union, Any

ITEM_SEPARATOR = f"{cn.COMMA} "
SUBITEM_SEPARATOR = "#"
STRUCTURE_SEPARATOR = "##"
SUPPORTED_TYPES = [bool, str, int, float, np.int64, np.ndarray, AssignmentPair, AssignmentCollection]


class CSVMaker:
    def __init__(self, type_dct:dict[str, type]) -> None:
        """
        Args:
            pairs (List[tuple[str, type]]): name, type
        """
        self.type_dct = dict(type_dct)
        self.names = list(type_dct.keys())
        for ttype in type_dct.values():
            if not ttype in SUPPORTED_TYPES:
                raise RuntimeError(f'{str(ttype)} is not a supported type!')

    def __repr__(self)->str:
        return self.makeHeader()
    
    def __eq__(self, other)->bool:
        return self.type_dct == other

    def makeHeader(self) -> str:
        return ", ".join(['"' + n + '"' for n in self.names])
    
    def append(self, other:'CSVMaker')->'CSVMaker':
        """
        Creates a new CSVMake that is the concatentation of this followed by other.

        Args:
            other (CSVMaker)

        Returns:
            CSVMaker
        """
        type_dct = dict(self.type_dct)
        type_dct.update(other.type_dct)
        return CSVMaker(type_dct)
    
    def encode(self, **kwargs) -> str:
        """
        Encodes the values as strings

        Args:
            kwargs: dict
                key: name
                value: List[Union[str, int, float, np.int64, np.ndarray]]

        Returns:
            str: _description_
        """
        items = []
        for name, value in kwargs.items():
            if self.type_dct[name] == np.ndarray:
                items.append(self._encodeArray(value))
            elif self.type_dct[name] == AssignmentPair:
                items.append(self._encodeAssignmentPair(value))
            elif self.type_dct[name] == AssignmentCollection:
                items.append(self._encodeAssignmentCollection(value))
            else:
                items.append(f'{value}')
        return ITEM_SEPARATOR.join(items)
    
    def decode(self, encoded_str:str)->dict:
        """Decodes the string into values of the correct type

        Args:
            encoded_str (str)

        Returns:
            dict:
                key: name
                value: value
        """
        items = encoded_str.split(ITEM_SEPARATOR)
        dct:dict[str, Any] = {}
        for name, item in zip(self.names, items):
            if self.type_dct[name] == np.ndarray:
                dct[name] = self._decodeArray(item)
            elif self.type_dct[name] == AssignmentPair:
                dct[name] = self._decodeAssignmentPair(item)  # type: ignore
            elif self.type_dct[name] == AssignmentCollection:
                dct[name] = self._decodeAssignmentCollection(item)
            elif self.type_dct[name] == bool:
                if item == 'True':
                    dct[name] = True
                else:
                    dct[name] = False
            else:
                dct[name] = self.type_dct[name](item)
        #
        return dct
    
    @staticmethod
    def _encodeArray(array:np.ndarray)->str:
        """
        Encodes an array as a string

        Args:
            array (np.ndarray): _description_

        Raises:
            ValueError: _description_

        Returns:
            Tuple[str, int, int]: _description_
        """
        array = np.array(array)
        if array.ndim == 1:
            num_column = len(array)
            num_row = 1
        elif array.ndim == 2:
            num_row, num_column = np.shape(array)
        else:
            raise ValueError("Array must be 1 or 2 dimensional.")
        flat_array = np.reshape(array, num_row*num_column)
        str_arr = [str(i) for i in flat_array]
        array_str = "[" + ",".join(str_arr) + "]"
        encoding = array_str + SUBITEM_SEPARATOR + str(num_row) + SUBITEM_SEPARATOR + str(num_column)
        return encoding

    @staticmethod
    def _decodeArray(encoded_arr:str)->np.ndarray:
        array_str, num_row, num_column = encoded_arr.split(SUBITEM_SEPARATOR)
        array = np.array(eval(array_str))
        array = np.reshape(array, (int(num_row), int(num_column)))
        return array
    
    @classmethod
    def _encodeAssignmentPair(cls, assignment_pair:AssignmentPair)->str:
        """
        Represents the assignment pair as two arrays.

        Args:
            assignment_pair (AssignmentPair)

        Returns:
            str
        """
        species_assignment_encoding = cls._encodeArray(assignment_pair.species_assignment)
        reaction_assignment_encoding = cls._encodeArray(assignment_pair.reaction_assignment)
        encoding = species_assignment_encoding + SUBITEM_SEPARATOR + reaction_assignment_encoding
        return encoding
    
    @classmethod
    def _decodeAssignmentPair(cls, encoding:str)->Union[None, AssignmentPair]:
        """
        Represents the assignment pair as two arrays.

        Args:
            encoding of an assignment pair: str

        Returns:
            AssignmentPair
        """
        pos = 0
        for _ in range(3):
            idx = encoding[pos+1:].find(SUBITEM_SEPARATOR)
            if idx < 0:
                return None
            pos += idx + 1
        species_assignment_encoding = encoding[0:pos]
        reaction_assignment_encoding = encoding[pos+1:]
        species_assignment = cls._decodeArray(species_assignment_encoding)
        reaction_assignment = cls._decodeArray(reaction_assignment_encoding)
        return AssignmentPair(species_assignment=species_assignment, reaction_assignment=reaction_assignment)

    @classmethod
    def _encodeAssignmentCollection(cls, assignment_collection:AssignmentCollection)->str:
        """
        Represents the assignment pair as two arrays.

        Args:
            assignment_pairs (List[AssignmentPair])

        Returns:
            str
        """
        encodings = []
        for assignment_pair in assignment_collection.pairs:
            encodings.append(cls._encodeAssignmentPair(assignment_pair))
        if (None in encodings) or (len(encodings) == 0):
            return "None"
        else:
            return STRUCTURE_SEPARATOR.join(encodings)
    
    @classmethod
    def _decodeAssignmentCollection(cls, encoding:str)->AssignmentCollection:
        """
        Represents the assignment pair as two arrays.

        Args:
            encoding of an assignment pair: str

        Returns:
            List[AssignmentPair]
        """
        if encoding == "None":
            return AssignmentCollection([])
        encodings = encoding.split(STRUCTURE_SEPARATOR)
        lst = [cls._decodeAssignmentPair(encoding) for encoding in encodings]
        if None in lst:
            return AssignmentCollection([])
        assignment_pairs = AssignmentCollection(lst)
        return assignment_pairs
