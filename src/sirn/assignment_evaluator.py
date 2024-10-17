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
import sirn.constants as cn  # type: ignore

import numpy as np
import time
from typing import Union, Tuple, Optional, List


DEBUG = False


#########################################################
class Timer(object):

    def __init__(self, name:str):
        self.name = name
        self.base_time = time.time()
        self.dct:dict = {}

    def add(self, name:str):
        self.dct[name] = time.time() - self.base_time
        self.base_time = time.time()

    def report(self):
        print(f"***{self.name}***")
        for key, value in self.dct.items():
            print(f"  {key}: {value}")
        print("************")


#########################################################
class AssignmentBatchContext(object):
    # Saves context for a batch of assignment evaluation
    def __init__(self, big_reference_arr, big_target_arr, big_target_row_idx_arr, big_target_column_idx_arr):
        self.big_reference_arr = big_reference_arr
        self.big_target_arr = big_target_arr
        self.big_target_row_idx_arr = big_target_row_idx_arr  # Indices for target rows
        self.big_target_column_idx_arr = big_target_column_idx_arr  # Indices for target columns



#########################################################
class _Assignment(object):
     
    def __init__(self, array:np.ndarray):
        self.array = array
        self.num_row, self.num_column = array.shape

    def __repr__(self)->str:
        return f"Assignment({self.array})"


#########################################################
class AssignmentEvaluator(object):
       
    def __init__(self, reference_arr:np.ndarray, target_arr:np.ndarray, max_batch_size:int=cn.MAX_BATCH_SIZE):
        """
        Args:
            reference_arr (np.ndarray): reference matrix
            target_arr (np.ndarray): target matrix
            row_assignment_arr (np.ndarray): candidate assignments of target rows to reference rows
            column_assignment_arr (np.ndarray): candidate assignments of target columns to reference columns
            comparison_criteria (ComparisonCriteria): comparison criteria
            max_batch_size (int): maximum batch size in units of bytes
        """
        self.reference_arr = reference_arr
        self.num_reference_row, self.num_reference_column = self.reference_arr.shape
        self.target_arr = target_arr   # Reduce memory usage
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
    
    def _experimentalMakeBatch(self, start_idx:int, end_idx:int, row_assignment:_Assignment, column_assignment:_Assignment,
          batch_context:Optional[AssignmentBatchContext]=None)->AssignmentBatchContext:
        """
        Constructs the reference and target matrices for a batch of comparisons. The approach is:
        1. Construct a flattened version of the target matrix for each comparison so that elements are ordered by
            row, then column, then comparison instance.

        Args:
            start_idx (int): start index
            end_idx (int): end index
            row_assignment (Assignment): row assignments
            column_assignment (Assignment): column assignments
            batch_context (_BatchContext): variables saved across invocations of _makeBatch

        Returns:
            AssignmentBatchContext
        """
        timer = Timer("makeBatch")
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
        timer.add("initializations")
        # Calculate the index of rows for the flattened target matrix. There is a row value for each
        #   comparison and element of the reference matrix.
        #      Index of the target rows
        row1_idx_arr = row_assignment.array[row_assignment_sel_arr].flatten()
        #      Replicate each index for the number of columns
        timer.add("row-1")
        # FIXME: This is the second slowest part of the code
        row2_idx_arr = np.repeat(row1_idx_arr, self.num_reference_column)
        timer.add("row-2")
        # Calculate the column for each element of the target flattened matrix
        column1_idx_arr = column_assignment.array[column_assignment_sel_arr]
        timer.add("column-1")
        column_idx_arr = np.repeat(column1_idx_arr, self.num_reference_row, axis=0).flatten()
        timer.add("column-2")
        if DEBUG:
            assert(len(row2_idx_arr) == len(column_idx_arr))
        # Construct the selected parts of the target matrix
        # FIXME: This is the slowest part of the code. Make empty flattened_target_arr and fill it
        if batch_context is None:
            # Reference array
            big_references = [self.reference_arr.flatten()]*num_comparison
            flattened_big_reference_arr = np.concatenate(big_references, axis=0)
            big_reference_arr = np.reshape(flattened_big_reference_arr,
                (self.num_reference_row*num_comparison, self.num_reference_column))
            big_reference_arr = big_reference_arr.astype(self.reference_arr.dtype)
            # Target array
            num_big_target_row = self.num_reference_row*num_comparison
            num_big_target_column = self.num_reference_column
            big_target_column_idx_arr = np.vstack([range(num_big_target_column)]*num_big_target_row)
            big_target_column_idx_arr = big_target_column_idx_arr.flatten()
            big_target_row_idx_arr = np.repeat(range(self.num_reference_row), num_big_target_column*num_comparison)
            big_target_arr = np.empty((num_big_target_row, num_big_target_column)).astype(self.target_arr.dtype)
            batch_context = AssignmentBatchContext(big_reference_arr=big_reference_arr,
                  big_target_arr=big_target_arr,
                  big_target_row_idx_arr=big_target_row_idx_arr,
                  big_target_column_idx_arr=big_target_column_idx_arr)
            timer.add("batch_context is None")
        batch_context.big_target_arr[batch_context.big_target_row_idx_arr, batch_context.big_target_column_idx_arr] =  \
            self.target_arr[row2_idx_arr, column_idx_arr]
        timer.add("Assign target")
        timer.report()
        #
        return batch_context
    
    def evaluateTarget(self, row_assignment_arr: np.ndarray, column_assignment_arr:np.ndarray,
              max_num_assignment=cn.MAX_NUM_ASSIGNMENT)->List[AssignmentPair]:
        """Evaluates the assignments for the target matrix.

        Args:
            row_assignment_arr (np.ndarray): assignments of target rows to reference rows
            column_assignment_arr (np.ndarray): assignments of target columns to reference columns

        Returns:
            AssignmentPair: Row and column assignments that result in equality
        """
        assignment_pairs = []
        num_assignment = row_assignment_arr.shape[0]*column_assignment_arr.shape[0]
        if num_assignment > max_num_assignment:
            frac_accept = max_num_assignment/num_assignment
        else:
            frac_accept = 1.0
        max_num_assignment = min(max_num_assignment, num_assignment)
        random_numbers = np.random.rand(int(max_num_assignment))
        for idx, row in enumerate(row_assignment_arr):
            if frac_accept < random_numbers[idx]:
                continue
            target_row_perm_arr = self.target_arr[row, :]
            for column in column_assignment_arr:
                target_arr = target_row_perm_arr[:, column]
                if np.all(self.reference_arr == target_arr):
                    assignment_pairs.append(AssignmentPair(row_assignment=row, column_assignment=column))
        return assignment_pairs
    
    def _compare(self, big_reference_arr:np.ndarray, big_target_arr:np.ndarray, is_close:bool)->np.ndarray:
        """
        Compares the reference matrix to the target matrix using a vectorized approach.

        Args:
            big_reference_arr: np.ndarray - reference stoichiometry matrix
            big_target_arr: np.ndarray  - target stoichiometry matrix
            is_close: bool - True if the comparison uses np.isclose
        
        Returns:
            np.ndarray[bool]: indicates successful (True) or unsuccessful (False) comparisons by assignment pair index
        """
        # Initializations
        num_big_row, num_big_column = big_reference_arr.shape
        if is_close:
            big_compatible_arr = np.isclose(big_reference_arr, big_target_arr)
        else:
            big_compatible_arr = big_reference_arr == big_target_arr
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
        Uses equality in comparisons (for speed) since it is expected that array elements
        will be 1 byte integers.

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
        #print("***num_batch", num_batch, "num_row_assignment", num_row_assignment, "num_column_assignment", num_column_assignment)
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
            assignment_satisfy_arr = self._compare(big_reference_arr, big_target_arr,
                  is_close=False)
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
        Uses np.isclose in comparisons

        Args:
            assignment_pairs: List[AssignmentPair] - pairs of row and column assignments

        Returns:
            List[AssignmentPair]: Assignment pairs that successfully compare
        """
        successful_assignment_pairs = []
        for assignment_pair in assignment_pairs:
            target_arr = self.target_arr[assignment_pair.row_assignment, :]
            target_arr = target_arr[:, assignment_pair.column_assignment]
            result = self._compare(self.reference_arr, target_arr, is_close=True)
            if result:
                successful_assignment_pairs.append(assignment_pair)
        return successful_assignment_pairs
