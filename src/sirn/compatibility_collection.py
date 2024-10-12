'''CompatibilityCollection consists of lists of rows in target that are compatible with a row in reference.'''


import sirn.constants as cn # type: ignore
from sirn.named_matrix import NamedMatrix # type: ignore

import copy
import itertools
import numpy as np
from typing import List, Tuple

NULL_NMAT = NamedMatrix(np.array([[]]))
NULL_INT = -1


class CompatibilityCollection(object):
    # A compatibility collection specifies the rows in self that are compatible with other.

    def __init__(self, num_self_row:int, num_other_row:int):
        self.num_self_row = num_self_row
        self.num_other_row = num_other_row
        self.compatibilities:list = [ [] for _ in range(num_self_row)]

    def add(self, reference_row:int, target_rows:List[int]):
        # Add rows in target that are compatible with a row in reference
        self.compatibilities[reference_row].extend(target_rows)

    def copy(self)->'CompatibilityCollection':
        new_collection = CompatibilityCollection(self.num_self_row, self.num_other_row)
        new_collection.compatibilities = [l.copy() for l in self.compatibilities]
        return new_collection

    def __repr__(self):
        return str(self.compatibilities)
    
    def __len__(self)->int:
        return len(self.compatibilities)

    def __eq__(self, other)->bool:
        if len(self.compatibilities) != len(other.compatibilities):
            return False
        trues = [np.all(self.compatibilities[i] == other.compatibilities[i]) for i in range(len(self.compatibilities))]
        return bool(np.all(trues))

    # FIXME: Use len(expand) 
    @property
    def log10_num_permutation(self)->float:
        # Calculates the log of the number of permutations implied by the compatibility collection
        lengths = [len(l) for l in self.compatibilities]
        if 0 in lengths:
            return -np.inf
        return np.sum([np.log10(len(l)) for l in self.compatibilities])

    def expand(self)->np.ndarray:
        MAX_BATCH_SIZE = 10000
        #####
        def selectUnique(array:np.ndarray)->np.ndarray:
            selection_arr = np.apply_along_axis(lambda x: len(x) == len(np.unique(x)), axis=1, arr=array)
            result = array[selection_arr, :]
            return result
        #####
        def expandCollection(collection:List[List[int]])->np.ndarray:
            # Expands the compatibilities into a two dimensional array where each row is a permutation
            candidate_arr = np.array(list(itertools.product(*collection)))
            return selectUnique(candidate_arr)
        #####
        def mergeAssignments(assignment1:np.ndarray, assignment2:np.ndarray)->np.ndarray:
            # Merges two assignments into a single assignment
            num_row1, num_row2 = assignment1.shape[0], assignment2.shape[0]
            assignment1 = np.repeat(assignment1, num_row2, axis=0)
            assignment2 = np.tile(assignment2, (num_row1, 1))
            merged_arr = np.concatenate([assignment1, assignment2], axis=1)
            return selectUnique(merged_arr)
        #####
        # Form batches of assignments no more than the maximum size and then merge the batches
        batches = []
        collection = copy.deepcopy(self.compatibilities)
        for _ in range(self.num_self_row):
            if len(collection) == 0:
                break
            # Construct a new batch
            num_assignment = 1
            assignment_idx = -1  # Index of the last assignment in the batch
            for idx in range(1, len(collection)):
                num_assignment *= len(collection[idx])
                if num_assignment > MAX_BATCH_SIZE:
                    break
                assignment_idx = idx
            batch_collection = collection[0:assignment_idx+1]
            if assignment_idx < len(collection) - 1:
                collection = collection[assignment_idx+1:]
            else:
                collection = []
            batch_assignment = expandCollection(batch_collection)
            batches.append(batch_assignment)
        # Merge the batches
        assignment_arr = batches[0]
        for idx in range(1, len(batches)):
            assignment_arr = mergeAssignments(assignment_arr, batches[idx])
        #
        return assignment_arr

    def prune(self, log10_max_permutation:float)->Tuple['CompatibilityCollection', bool]:
        """Randomly prune the compatibility collection to a maximum number of permutations

        Args:
            log10_max_permutation (float): log10 of the maximum number of permutations

        Returns:
            CompatibilityCollection
        """
        collection = self.copy()
        #
        is_changed = False
        for idx in range(1000000):
            if collection.log10_num_permutation <= log10_max_permutation:
                break
            candidate_rows = [i for i in range(collection.num_self_row)
                              if len(collection.compatibilities[i]) > 1]  
            idx = np.random.randint(0, len(candidate_rows))
            irow = candidate_rows[idx]
            if len(collection.compatibilities[irow]) <= 1:
                continue
            # Check for duplicate single values
            pos = np.random.randint(0, len(collection.compatibilities[irow]))
            singles = list(np.array([v for v in collection.compatibilities if len(v) == 1]).flatten())
            lst = collection.compatibilities[irow][0:pos]
            lst.extend(collection.compatibilities[irow][pos+1:])
            if (len(lst) == 1) and (lst[0] in singles):
                continue
            # Delete the element
            del collection.compatibilities[irow][pos]
            is_changed = True
        else:
            raise ValueError("Could not prune the collection.")
        #
        return collection, is_changed