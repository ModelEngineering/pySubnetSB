'''Compares the reference matrix with selected rows and columns in a target matrix.'''

"""
    Compares the reference matrix with selected rows and columns in a target matrix. The comparison is
    vectorized and checks if the reference matrix is identical to the target matrix when the rows and columns
    are permuted according to the selected assignments. The comparison is for equality, numerical inequality,
    or bitwise inequality.

    The comparison is performed by constructing a large matrix that is a concatenation of the reference matrix
    and the target matrix. The large matrix is constructed by repeating the reference matrix for each assignment
    of target rows to reference rows and target columns to reference columns. The target matrix is re-arranged
    according to the assignments. The large matrix is compared to the re-arranged target matrix. The comparison
    is performed for each assignment pair. The results are used to determine if the assignment pair results in
    an identical matrix.

    Some core concepts:
    * Assignment is a selection of a subset of rows (columns) in the target matrix to be compared to the reference matrix.
    * AssignmentPair is a pair of assignments of rows and columns in the target matrix to the reference matrix.
    * AssignmentPairIndex. An index into the cross-product of the row and column assignments. This is referred to
        as linear addressing. Using the separate indices for rows and columns is referred to as vector addressing.
        Methods are provided convert
        between the assignment pair index and the indices of row and column assignments.
    * AssignmentArray is a two dimensional array of assignments. Columns are the index of the target row that
        is assigned to the reference row (column index).
        Rows are instances of an assignment. There is an AssignmentArray for the rows and for the columns.
    * A Comparison is a comparison betweeen the reference matrix and an assignment of rows and columns in the target

 """
from sirn.assignment_pair import AssignmentPair # type: ignore

import numpy as np
from typing import Union, Tuple, Optional, List


MAX_BATCH_SIZE = int(1e7)  # Matrix memory in bytes used in a comparison batch
DEBUG = False


#########################################################
class _Assignment(object):
     
    def __init__(self, array:np.ndarray):
        self.array = array
        self.num_row, self.num_column = array.shape

    def __repr__(self)->str:
        return f"Assignment({self.array})"
    

#########################################################
class ComparisonCriteria(object):
    # Describes the comparisons to be made between the reference and target matrices. Choices are mutually exclusive.

    def __init__(self, is_equality:bool=False, is_numerical_inequality:bool=False,
            is_bitwise_inequality:bool=False):
            """
            Args:
                is_equality (bool): if True, the comparison is for equality
                is_numerical_inequality (bool): if True, the comparison is for numerical inequality
                is_bitwise_inequality (bool): if True, the comparison is for bitwise inequality
            """
            if is_equality + is_numerical_inequality + is_bitwise_inequality != 1:
                raise ValueError("Must specify exactly one comparison criteria")
            self.is_equality = is_equality
            self.is_numerical_inequality = is_numerical_inequality
            self.is_bitwise_inequality = is_bitwise_inequality

    def __repr__(self)->str:
        return f"ComparisonCriteria({self.is_equality}, {self.is_numerical_inequality}, {self.is_bitwise_inequality})"


#########################################################
class AssignmentEvaluator(object):
       
    def __init__(self, reference_arr:np.ndarray, target_arr:np.ndarray, comparison_criteria:ComparisonCriteria,
              max_batch_size:int=MAX_BATCH_SIZE):
        """
        Args:
            reference_arr (np.ndarray): reference matrix
            target_arr (np.ndarray): target matrix
            row_assignment_arr (np.ndarray): candidate assignments of target rows to reference rows
            column_assignment_arr (np.ndarray): candidate assignments of target columns to reference columns
            comparison_criteria (ComparisonCriteria): comparison criteria
            max_batch_size (int): maximum batch size in units of bytes
        """
        self.reference_arr = reference_arr.astype(np.int16)  # Reduce memory usage
        self.num_reference_row, self.num_reference_column = self.reference_arr.shape
        self.target_arr = target_arr.astype(np.int16)   # Reduce memory usage
        self.comparison_criteria = comparison_criteria
        self.max_batch_size = max_batch_size

    def vectorToLinear(self, num_column_assignment:int, row_idx:Union[int, np.ndarray[int]],  # type: ignore
          column_idx:Union[int, np.ndarray[int]])->Union[int, np.ndarray[int]]:  # type: ignore
        """
        Converts a vector address of the candidate pair assignments (row, column) to a linear address.
        The linearlization is column first.

        Args:
            num_column_assignment (int): number of column assignments
            row_idx (int): row index
            column_idx (int): column index

        Returns:
            np.ndarray: linear index
        """
        return row_idx*num_column_assignment + column_idx

    def linearToVector(self, num_column_assignment:int,
          index:Union[int, np.ndarray])->Union[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:  
        """
        Converts a linear address of the candidate assignments to a vector index.
        Converts a vector representation of the candidate assignments (row, column) to a linear index

        Args:
            num_column_assignment (int): number of column assignments
            index (int): linear index

        Returns:
            row_idx (int): row index
            column_idx (int): column index
        """
        row_idx = index//num_column_assignment
        column_idx = index%num_column_assignment
        return row_idx, column_idx  # type: ignore
    
    def _makeBatch(self, start_idx:int, end_idx:int, row_assignment:_Assignment, column_assignment:_Assignment,
          big_reference_arr:Optional[np.ndarray]=None)->Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the reference and target matrices for a batch of comparisons. The approach is:
        1. Construct a flattened version of the target matrix for each comparison so that elements are ordered by
            row, then column, then comparison instance.

        Args:
            start_idx (int): start index
            end_idx (int): end index
            row_assignment (Assignment): row assignments
            column_assignment (Assignment): column assignments
            big_reference_arr (np.ndarray): big reference array

        Returns:
            np.ndarray: big_reference_arr
            np.ndarray: big_target_arr
        """
        num_comparison = end_idx - start_idx + 1
        if num_comparison > row_assignment.num_row*column_assignment.num_row:
            raise ValueError("Number of comparisons exceeds the number of assignments")
        num_column_assignment = column_assignment.num_row
        row_assignment_sel_arr, column_assignment_sel_arr = self.linearToVector(
              num_column_assignment, np.array(range(start_idx, end_idx+1)))
        if row_assignment_sel_arr.shape[0] != num_comparison:  # type: ignore
            raise ValueError("Number of comparisons does not match the indices.")
        if column_assignment_sel_arr.shape[0] != num_comparison:  # type: ignore
            raise ValueError("Number of comparisons does not match the indices.")
        # Calculate the index of rows for the flattened target matrix. There is a row value for each
        #   comparison and element of the reference matrix.
        #      Index of the target rows
        row1_idx_arr = row_assignment.array[row_assignment_sel_arr].flatten()
        #      Replicate each index for the number of columns
        row2_idx_arr = np.repeat(row1_idx_arr, self.num_reference_column)
        # Calculate the column for each element of the target flattened matrix
        column1_idx_arr = column_assignment.array[column_assignment_sel_arr]
        column_idx_arr = np.repeat(column1_idx_arr, self.num_reference_row, axis=0).flatten()
        if DEBUG:
            assert(len(row2_idx_arr) == len(column_idx_arr))
        # Construct the selected parts of the target matrix
        flattened_target_arr = self.target_arr[row2_idx_arr, column_idx_arr]
        big_target_arr = np.reshape(flattened_target_arr,
              (self.num_reference_row*num_comparison, self.num_reference_column))
        if DEBUG:
            assert(big_target_arr.shape[0] == self.num_reference_row*num_comparison)
        # Construct the reference array
        if big_reference_arr is None:
            big_references = [self.reference_arr.flatten()]*num_comparison
            flattened_big_reference_arr = np.concatenate(big_references, axis=0)
            big_reference_arr = np.reshape(flattened_big_reference_arr,
                (self.num_reference_row*num_comparison, self.num_reference_column))
        return big_reference_arr, big_target_arr
    
    def _compare(self, big_reference_arr:np.ndarray, big_target_arr:np.ndarray)->np.ndarray:
        """
        Compares the reference matrix to the target matrix using a vectorized approach.

        Args:
            big_reference_arr: np.ndarray - reference stoichiometry matrix
            big_target_arr: np.ndarray  - target stoichiometry matrix
        
        Returns:
            np.ndarray[bool]: indicates successful (True) or unsuccessful (False) comparisons by assignment pair index
        """
        # Initializations
        num_big_row, num_big_column = big_reference_arr.shape
        # Do the comparisons
        if self.comparison_criteria.is_equality:
            big_compatible_arr = big_reference_arr == big_target_arr
        elif self.comparison_criteria.is_numerical_inequality:
            big_compatible_arr = big_reference_arr <= big_target_arr
        elif self.comparison_criteria.is_bitwise_inequality:
            big_compatible_arr = big_reference_arr & big_target_arr == big_reference_arr
        else:
            raise RuntimeError("Unknown comparison criteria.")
        # Determine the successful assignment pairs
        big_row_sum = np.sum(big_compatible_arr, axis=1)
        big_row_satisfy = big_row_sum == num_big_column # Sucessful row comparisons
        num_assignment_pair = num_big_row//self.num_reference_row
        assignment_pair_satisfy_arr = np.reshape(big_row_satisfy, (num_assignment_pair, self.num_reference_row))
        assignment_satisfy_arr = np.sum(assignment_pair_satisfy_arr, axis=1) == self.num_reference_row
        #
        return assignment_satisfy_arr

    def evaluateAssignmentArrays(self, row_assignment_arr:np.ndarray,
          column_assignment_arr:np.ndarray)->List[AssignmentPair]:
        """Finds the row and column assignments that satisfy the comparison criteria.

        Args:
            row_assignment_arr (np.ndarray): assignments of target rows to reference rows
            column_assignment_arr (np.ndarray): assignments of target columns to reference columns

        Returns:
            List[AssignmentPair]: pairs of row and column assignments that satisfy the comparison criteria
        """
        # Initializations
        num_row_assignment = row_assignment_arr.shape[0]
        num_column_assignment = column_assignment_arr.shape[0]
        num_comparison = num_row_assignment*num_column_assignment
        row_assignment = _Assignment(row_assignment_arr)
        column_assignment = _Assignment(column_assignment_arr)
        # Error checks
        if row_assignment.num_column != self.num_reference_row:
            raise ValueError("Number of reference rows does not match the number of row assignments")
        if column_assignment.num_column != self.num_reference_column:
            raise ValueError("Number of reference columns does not match the number of row assignments")
        # Calculate the number of assignment pair indices in a batch
        bytes_per_comparison = 2*self.reference_arr.itemsize*self.num_reference_row*self.num_reference_column
        total_bytes = bytes_per_comparison*num_comparison
        max_comparison_per_batch = max(self.max_batch_size//bytes_per_comparison, 1)
        num_batch = num_comparison//max_comparison_per_batch + 1
        # Iterative do the assignments
        assignment_pairs = []
        big_reference_arr = None
        for ibatch in range(num_batch):
            start_idx = ibatch*max_comparison_per_batch
            if start_idx >= num_comparison:
                continue
            end_idx = min((ibatch+1)*max_comparison_per_batch, num_comparison - 1)
            if end_idx == num_comparison - 1:
                # Last batch must adjust the size of the reference array
                big_reference_arr = None
            big_reference_arr, big_target_arr = self._makeBatch(start_idx, end_idx, row_assignment, column_assignment,
                  big_reference_arr=big_reference_arr)
            assignment_satisfy_arr = self._compare(big_reference_arr, big_target_arr)
            assignment_satisfy_idx = np.where(assignment_satisfy_arr)[0]
            # Add the assignment pairs
            for idx in assignment_satisfy_idx:
                adjusted_idx = idx + start_idx
                row_idx, column_idx = self.linearToVector(num_column_assignment, adjusted_idx)
                assignment_pair = AssignmentPair(row_assignment=row_assignment_arr[row_idx],
                      column_assignment=column_assignment_arr[column_idx])
                assignment_pairs.append(assignment_pair)
        return assignment_pairs

    def evaluateAssignmentPairs(self, assignment_pairs:List[AssignmentPair])->List[AssignmentPair]:
        """Finds the pair of row and column assignments that satsify the comparison criteria.

        Args:
            assignment_pairs (Optional[List[AssignmentPair]], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            List[AssignmentPair]: _description_
        """
        raise NotImplementedError("Must implement.")