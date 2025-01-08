'''CompatibilityCollection consists of lists of rows in target that are compatible with a row in reference.'''


import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.named_matrix import NamedMatrix # type: ignore
from pySubnetSB import util # type: ignore

import copy
import itertools
import numpy as np
from typing import List, Tuple

NULL_NMAT = NamedMatrix(np.array([[]]))
NULL_INT = -1
NUM_SAMPLE = 1000  # Number of samples to estimate the number of assignments


class CompatibilityCollection(object):
    # A compatibility collection specifies the rows in self that are compatible with other.

    def __init__(self, num_self_row:int, num_other_row:int):
        self.num_self_row = num_self_row
        self.num_other_row = num_other_row
        self.compatibilities:list = [ [] for _ in range(num_self_row)]

    @classmethod
    def makeFromListOfLists(cls, list_of_lists:List[List[int]])->'CompatibilityCollection':
        num_self_row = len(list_of_lists)
        num_other_row = np.max([max(l) for l in list_of_lists])
        collection = CompatibilityCollection(num_self_row, num_other_row)
        collection.compatibilities = list_of_lists
        return collection

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
    
    def __le__(self, other)->bool:
        # self is less equal than other
        NUM_CHECK = 20
        # Allow for some variation because of random sampling of the compatibility collection
        comparisons = [other.log10_num_assignment - self.log10_num_assignment > -0.5
              for _ in range(NUM_CHECK)]
        return np.sum(comparisons) > NUM_CHECK*0.75

    @property
    def log10_num_assignment(self)->float:
        # Calculates the log of the number of assignments implied by the compatibility collection
        lengths = [len(l) for l in self.compatibilities]
        if 0 in lengths:
            return -np.inf
        unadjusted_estimate = np.sum([np.log10(len(l)) for l in self.compatibilities])
        # Estimate the number of assignments with duplicate indices
        sample_arr = util.sampleListOfLists(self.compatibilities, NUM_SAMPLE)
        unique_sample_arr = self._selectUnique(sample_arr)
        frac_unique = max(unique_sample_arr.shape[0] / sample_arr.shape[0], 1e-9)
        adjusted_estimate = max(0, unadjusted_estimate + np.log10(frac_unique))
        return adjusted_estimate

    @staticmethod
    def _selectUnique(array:np.ndarray)->np.ndarray:
        # Prunes rows in which there are duplicate values
        if len(array) > 0:
            idxs = np.all(np.diff(np.sort(array), axis=1) != 0, axis=1)
            result = array[idxs]
        else:
            result = np.array([])
        return result

    def expand(self, max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT)->Tuple[np.ndarray, bool]:
        """Expands the compatibility collection into a two dimensional array of assignments

        Args:
            max_num_assignment (int, optional): maximum number of assignments Defaults to cn.MAX_NUM_ASSIGNMENT.

        Returns:
            np.ndarray: two dimensional array of assignments
            bool: is truncated
        """
        MAX_BATCH_SIZE = 1000
        #####
        def expandCollection(collection:List[List[int]])->np.ndarray:
            # Expands the compatibilities into a two dimensional array where each row is an assignment
            candidate_arr = np.array(list(itertools.product(*collection)))
            return self._selectUnique(candidate_arr)
        #####
        def mergeAssignments(assignment1:np.ndarray, assignment2:np.ndarray)->np.ndarray:
            # Merges two assignments into a single assignment
            # Check for degenerate conditions
            if assignment1.shape[0] == 0:
                return self._selectUnique(assignment2)
            if assignment2.shape[0] == 0:
                return self._selectUnique(assignment1)
            # Merge the assignments
            num_row1, num_row2 = assignment1.shape[0], assignment2.shape[0]
            big_assignment1 = np.repeat(assignment1, num_row2, axis=0)
            big_assignment2 = np.vstack([assignment2]*num_row1)
            merged_arr = np.concatenate([big_assignment1, big_assignment2], axis=1)
            result = self._selectUnique(merged_arr)
            return result
        #####
        # Form batches of assignments no more than the maximum size and then merge the batches
        log10_max_num_assignment = np.log10(max_num_assignment)
        collection, is_truncated = self.prune(log10_max_num_assignment=log10_max_num_assignment)
        list_of_lists = collection.compatibilities
        batches = []
        for _ in range(self.num_self_row):
            if len(list_of_lists) == 0:
                break
            # Construct a new batch
            num_assignment = 1
            assignment_idx = -1  # Index of the last assignment in the batch
            for idx in range(len(list_of_lists)):
                num_assignment *= len(list_of_lists[idx])
                if num_assignment > MAX_BATCH_SIZE:
                    break
                assignment_idx = idx
            #assert(assignment_idx >= 0)
            batch_collection = list_of_lists[0:assignment_idx+1]
            if assignment_idx < len(list_of_lists):
                list_of_lists = list_of_lists[assignment_idx+1:]
            else:
                list_of_lists = []
            expanded_collection = expandCollection(list(batch_collection))
            if expanded_collection.shape[0] == 0:
                return np.array([]), True
            batches.append(expandCollection(list(batch_collection)))
        # Merge the batches
        assignment_arr = batches[0]
        for idx in range(1, len(batches)):
            assignment_arr = mergeAssignments(assignment_arr, batches[idx])
        #
        #assert(assignment_arr.shape[1] == len(self.compatibilities))
        return assignment_arr, is_truncated

    def prune(self, log10_max_num_assignment:float)->Tuple['CompatibilityCollection', bool]:
        """Randomly prune the compatibility collection to a maximum number of assignments

        Args:
            log10_max_assignment (float): log10 of the maximum number of assignments

        Returns:
            CompatibilityCollection
            bool: is changed
        """
        collection = self.copy()
        #
        is_changed = False
        for idx in range(1000000):
            if collection.log10_num_assignment <= log10_max_num_assignment:
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