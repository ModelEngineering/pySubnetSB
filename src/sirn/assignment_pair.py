'''Pairs of species and reaction assignments.'''

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
        return AssignmentPair(species_assignment=self.species_assignment, reaction_assignment=self.reaction_assignment)
    
    def __eq__(self, other)->bool:
        if not isinstance(other, AssignmentPair):
            return False
        if not np.all(self.species_assignment == other.species_assignment):
            return False
        if not np.all(self.reaction_assignment == other.reaction_assignment):
            return False
        return True


class AssignmentCollection(object):

    def __init__(self, assignment_collection:List[AssignmentPair]):
        self.pairs = assignment_collection

    def __eq__(self, other):
        if not isinstance(other, AssignmentCollection):
            return False
        return all([a1 == a2 for a1, a2 in zip(self.pairs, other.pairs)])
