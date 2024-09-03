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

    def copy(self)->'AssignmentPair':
        return AssignmentPair(species_assignment=self.species_assignment.copy(),
                              reaction_assignment=self.reaction_assignment.copy())
    
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
        return json.dumps({cn.S_ID: str(self.__class__),
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
        if not str(cls) in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        species_assignment = np.array(dct[cn.S_SPECIES_ASSIGNMENT_LST])
        reaction_assignment = np.array(dct[cn.S_REACTION_ASSIGNMENT_LST])
        return AssignmentPair(species_assignment=species_assignment, reaction_assignment=reaction_assignment)


class AssignmentCollection(object):

    def __init__(self, assignment_collection:List[AssignmentPair]):
        self.pairs = assignment_collection

    def __eq__(self, other):
        if not isinstance(other, AssignmentCollection):
            return False
        return all([a1 == a2 for a1, a2 in zip(self.pairs, other.pairs)])
