'''Pairs of species and reaction assignments.'''

import pySubnetSB.constants as cn  # type: ignore
import pySubnetSB.util as util  # type: ignore

import json
import numpy as np  # type: ignore
from typing import List, Optional

class AssignmentPair(object):
    # species and row are equivalent. reaction and column are equivalent.

    def __init__(self, species_assignment=None, reaction_assignment=None, row_assignment=None, column_assignment=None,
          reference_species_names:Optional[List[str]]=None,
          target_species_names:Optional[List[str]]=None,
          reference_reaction_names:Optional[List[str]]=None,
          target_reaction_names:Optional[List[str]]=None):
        """
        Assignment of species and reactions to target to describe its permutation into the reference.

        Args:
            species_assignment (_type_, optional): Assignment of target species (values) to reference (position)
            reaction_assignment (_type_, optional): Assignment of target reactions (values) to reference (position)
            row_assignment (_type_, optional): Same as species
            column_assignment (_type_, optional): Same as reactions
            reference_species_names (Optional[List[str]], optional)
            target_species_names (Optional[List[str]], optional)
            reference_reaction_names (Optional[List[str]], optional)
            target_reaction_names (Optional[List[str]], optional)

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
        """
        if (species_assignment is None) and (row_assignment is None):
            raise RuntimeError("Must specify species assignment!")
        if (reaction_assignment is None) and (column_assignment is None):
            raise RuntimeError("Must specify reaction assignment!")
        self._species_assignment = species_assignment
        self._reaction_assignment = reaction_assignment
        self._row_assignment = row_assignment
        self._column_assignment = column_assignment
        self.reference_species_names = reference_species_names
        self.reference_reaction_names = reference_reaction_names
        self.target_species_names = target_species_names
        self.target_reaction_names = target_reaction_names

    @property
    def species_assignment(self)->np.ndarray:
        if self._species_assignment is None:
            self._species_assignment = self._row_assignment
        return self._species_assignment
    
    @property
    def row_assignment(self)->np.ndarray:
        if self._row_assignment is None:
            self._row_assignment = self._species_assignment
        return self._row_assignment
    
    @property
    def reaction_assignment(self)->np.ndarray:
        if self._reaction_assignment is None:
            self._reaction_assignment = self._column_assignment
        return self._reaction_assignment
    
    @property
    def column_assignment(self)->np.ndarray:
        if self._column_assignment is None:
            self._column_assignment = self._reaction_assignment
        return self._column_assignment

    def __repr__(self):
        return f"species: {self.species_assignment}, reaction: {self.reaction_assignment}"

    def copy(self)->'AssignmentPair':
        return AssignmentPair(species_assignment=self.species_assignment.copy(),
                              reaction_assignment=self.reaction_assignment.copy())
    
    def invert(self)->'AssignmentPair':
        """Invert the assignment pair.

        Returns:
            AssignmentPair
        """
        return AssignmentPair(species_assignment=np.argsort(self.species_assignment.copy()),
                              reaction_assignment=np.argsort(self.reaction_assignment.copy()))
    
    def __eq__(self, other)->bool:
        if not self.__class__.__name__ in str(type(other)):
            return False
        if (len(self.species_assignment) != len(other.species_assignment)):
            return False
        if (len(self.reaction_assignment) != len(other.reaction_assignment)):
            return False
        if not np.all(self.species_assignment == other.species_assignment):
            return False
        if not np.all(self.reaction_assignment == other.reaction_assignment):
            return False
        return True
    
    def serialize(self)->str:
        """Create a JSON string for the object.

        Returns:
            str: _description_
        """
        species_assignment_lst = self.species_assignment.tolist()
        reaction_assignment_lst = self.reaction_assignment.tolist()
        return json.dumps({cn.S_ID: self.__class__.__name__,
                           cn.S_SPECIES_ASSIGNMENT_LST: species_assignment_lst,
                           cn.S_REACTION_ASSIGNMENT_LST: reaction_assignment_lst})
    
    @classmethod
    def deserialize(cls, serialization_str:str)->'AssignmentPair':
        """Creates an AssignmentPair from its JSON string serialization:

        Args:
            serialization_str (str)

        Returns:
            AssignmentPair
        """
        dct = json.loads(serialization_str)
        if not cls.__name__ in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls.__name__} but got {dct[cn.S_ID]}")
        species_assignment = np.array(dct[cn.S_SPECIES_ASSIGNMENT_LST])
        reaction_assignment = np.array(dct[cn.S_REACTION_ASSIGNMENT_LST])
        return AssignmentPair(species_assignment=species_assignment, reaction_assignment=reaction_assignment)
    
    def resize(self, new_size)->'AssignmentPair':
        """Truncates the assignment pair to the specify size.

        Args:
            new_size (int): New size for the assignment pair

        Returns:

        """
        return AssignmentPair(species_assignment=self.species_assignment[:new_size],
                              reaction_assignment=self.reaction_assignment[:new_size])

    def makeNameDct(self)->dict:
        """Create a dictionary of names for the assignment of target species and reactions to the reference network.

        Returns:
            dict:
              keys: cn.SPECIES_NAMES, cn.REACTION_NAMES
              values: dict
                keys: reference name
                values: target name
        """
        if (self.reference_species_names is None) or (self.target_species_names is None):
            raise ValueError("Must specify species names.")
        if (self.reference_reaction_names is None) or (self.target_reaction_names is None):
            raise ValueError("Must specify reaction names.")
        species_dct = {self.reference_species_names[i]: self.target_species_names[self.species_assignment[i]]
                       for i in range(len(self.species_assignment))}
        reaction_dct = {self.reference_reaction_names[i]: self.target_reaction_names[self.reaction_assignment[i]]
                       for i in range(len(self.reaction_assignment))}
        return {cn.SPECIES_NAMES: species_dct, cn.REACTION_NAMES: reaction_dct}