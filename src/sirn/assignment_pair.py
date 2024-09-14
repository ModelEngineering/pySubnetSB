'''Pairs of species and reaction assignments.'''

import sirn.constants as cn  # type: ignore

import json
import numpy as np  # type: ignore
from typing import List

class AssignmentPair(object):

    def __init__(self, species_assignment=None, reaction_assignment=None):
        if species_assignment is None:
            raise RuntimeError("Must specify species assignment!")
        if reaction_assignment is None:
            raise RuntimeError("Must specify reaction assignment!")
        self.species_assignment = species_assignment
        self.reaction_assignment = reaction_assignment

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
        if not isinstance(other, AssignmentPair):
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