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
from pySubnetSB.assignment_pair import AssignmentPair # type: ignore
from pySubnetSB.assignment_evaluator_worker import AssignmentEvaluatorWorker, WorkerResult # type: ignore
from pySubnetSB import constants as cn # type: ignore
from pySubnetSB import util # type: ignore
from src.pySubnetSB.performance_monitor import PerformanceMonitor # type: ignore

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import multiprocessing as mp
from typing import Union, Tuple, Optional, List


DEBUG = False
IS_ENABLE_PERFORMANCE_EVALUATION = False
MIN_ASSIGNMENT_PER_PARALLEL_CPU = int(1e2)



#########################################################
class AssignmentBatchContext(object):
    # Saves context for a batch of assignment evaluation
    def __init__(self, big_reference_arr, big_target_arr, big_target_row_idx_arr, big_target_column_idx_arr):
        self.big_reference_arr = big_reference_arr
        self.big_target_arr = big_target_arr
        self.big_target_row_idx_arr = big_target_row_idx_arr  # Indices for target rows
        self.big_target_column_idx_arr = big_target_column_idx_arr  # Indices for target columns


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
        self.monitor = PerformanceMonitor("AssignmentEvaluator", is_enabled=IS_ENABLE_PERFORMANCE_EVALUATION)
        self.reference_arr = reference_arr
        self.num_reference_row, self.num_reference_column = self.reference_arr.shape
        self.target_arr = target_arr   # Reduce memory usage
        self.max_batch_size = max_batch_size

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

    #@profile
    def evaluateAssignmentPairs(self, assignment_pairs:List[AssignmentPair],
          max_num_assignment:int=-1)->List[AssignmentPair]:
        """Finds the pair of row and column assignments that satsify the comparison criteria.
        Uses np.isclose in comparisons

        Args:
            assignment_pairs: List[AssignmentPair] - pairs of row and column assignments
            max_num_assignment: int - maximum number of assignments (no bound if -1)

        Returns:
            List[AssignmentPair]: Assignment pairs that successfully compare
        """
        successful_assignment_pairs = []
        if max_num_assignment == -1:
            max_num_assignment = len(assignment_pairs)
        for assignment_pair in assignment_pairs:
            target_arr = self.target_arr[assignment_pair.row_assignment, :]
            target_arr = target_arr[:, assignment_pair.column_assignment]
            result = AssignmentEvaluatorWorker.compare(self.reference_arr, target_arr, self.num_reference_row,
                  is_close=True)
            if result:
                successful_assignment_pairs.append(assignment_pair)
            if len(successful_assignment_pairs) >= max_num_assignment:
                break
        return successful_assignment_pairs

    #@profile
    def parallelEvaluate(self, row_assignment_arr:np.ndarray, column_assignment_arr:np.ndarray,
                 total_process:int=-1, max_num_assignment:int=cn.MAX_NUM_ASSIGNMENT,
                 is_report:bool=True)->WorkerResult:
        """Evaluates the assignments for the target matrix using iteration.

        Args:
            row_assignment_arr (np.ndarray): assignments of target rows to reference rows
            column_assignment_arr (np.ndarray): assignments of target columns to reference columns
            num_process (int): number of processes
            total_process (int): number of processes (defaults to the number of CPUs)
            max_num_assignment (int): maximum number of assignments
            is_report (bool): report progress

        Returns:
            WorkerResult: results of the evaluation
                AssignmentPair: Row and column assignments that result in equality
                is_truncated: True if the evaluation was truncated
        """
        self.monitor.add("parallelEvaluate: Start")
        num_comparison = row_assignment_arr.shape[0]*column_assignment_arr.shape[0]
        # Initializations
        if total_process == -1:
            total_process = mp.cpu_count()
        num_process = min(mp.cpu_count(), total_process)
        if num_comparison < MIN_ASSIGNMENT_PER_PARALLEL_CPU:
            num_process = 1
        # Handle the case of a single process
        if num_process == 1:
            self.monitor.add("parallelEvaluate/Single Process: Start")
            # Construct the assignments
            row_assignments = [row_assignment_arr]
            column_assignments = [column_assignment_arr]
            # Run in a single process
            return_dct:dict = {}
            procnum = 0
            AssignmentEvaluatorWorker.do(self.reference_arr, self.target_arr, self.max_batch_size,
                  row_assignment_arr, column_assignment_arr, procnum, total_process,
                  return_dct, max_num_assignment,
                  is_report=is_report)
            self.monitor.add("parallelEvaluate/Single Process: End")
            return return_dct[0]
        else:
            self.monitor.add("parallelEvaluate/Multiple Process: Start")
            # Construct the assignments
            if row_assignment_arr.shape[0] > column_assignment_arr.shape[0]:
                row_assignments = util.partitionArray(row_assignment_arr, num_process)
                actual_num_process = len(row_assignments)
                column_assignments = [column_assignment_arr]
                is_row_max = True
            else:
                row_assignments = [row_assignment_arr]
                column_assignments = util.partitionArray(column_assignment_arr, num_process)
                actual_num_process = len(column_assignments)
                is_row_max = False
            self.monitor.add("parallelEvaluate/Multiple Process/Assignments: Start")
            # Construct the arguments
            args = []
            return_dct = {}
            for procnum in range(actual_num_process):
                if is_row_max:
                    process_row_assignment_arr = row_assignments[procnum]
                    process_column_assignment_arr = column_assignments[0]
                else:
                    process_row_assignment_arr = row_assignments[0]
                    process_column_assignment_arr = column_assignments[procnum]
                args.append((self.reference_arr, self.target_arr, self.max_batch_size,
                      process_row_assignment_arr, process_column_assignment_arr,
                      procnum, total_process, return_dct, max_num_assignment, is_report))
            self.monitor.add("parallelEvaluate/Multiple Process/Assignments: End")
            # Run the processes
            self.monitor.add("parallelEvaluate/Multiple Process/Run process pool: Start")
            with ProcessPoolExecutor(max_workers=total_process) as executor:
                process_args = zip(*args)
                results = executor.map(AssignmentEvaluatorWorker.do, *process_args)
            self.monitor.add("parallelEvaluate/Multiple Process/Run process pool: End")
        # Combine the results
        self.monitor.add("parallelEvaluate/Multiple Process/Combine results: Start")
        assignment_pairs = []
        is_truncated = False
        for result in results:
            is_truncated = is_truncated or result.is_truncated
            assignment_pairs.extend(result.assignment_pairs)
        worker_result = WorkerResult(assignment_pairs=assignment_pairs, is_truncated=is_truncated)
        self.monitor.add("parallelEvaluate/Multiple Process/Combine results: End")
        return worker_result
    
    def _manageEvaluation(self, reference_arr:np.ndarray, target_arr:np.ndarray, max_batch_size:int,
           row_assignment_arr:np.ndarray, column_assignment_arr:np.ndarray,
           procnum:int, total_proc:int)->List[AssignmentPair]:
        """Manages the evaluation of assignments.

        Args:
            reference_arr (np.ndarray): _description_
            target_arr (np.ndarray): _description_
            max_batch_size (int): _description_
            row_assignment_arr (np.ndarray): _description_
            column_assignment_arr (np.ndarray): _description_
            procnum (int): Number of this process
            total_proc (int): Number of processes

        Returns:
            List[AssignmentPair]: _description_
        """


        raise NotImplementedError("Subclass must implement this method")