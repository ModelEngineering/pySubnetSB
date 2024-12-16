'''Benchmarks for isStructurallyIdentical and isStructurallyIdenticalSubnet.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.network import Network # type: ignore
from pySubnetSB.assignment_pair import AssignmentPair  # type: ignore

import json
import numpy as np
import os
import shutil
import time
from typing import List


REFERENCE = "reference"
TARGET = "target"
S_RUNTIMES = "s_runtimes"
S_NUM_SUCCESS = "s_num_success"
S_NUM_TRUNCATED = "s_num_truncated"
S_IS_IDENTICALS = "s_is_identicals"
S_IS_TRUNCATEDS = "s_is_truncateds"
S_REFERENCE_SIZE = "s_reference_size"
S_EXPANSION_FACTOR = "s_expansion_factor"
S_NUM_EXPERIMENT = "s_num_experiment"
S_IS_IDENTICAL = "s_is_identical"
S_EXPERIMENTS = "s_experiments"
S_BENCHMARK_RUNNER = "s_benchmark_runner"


##############################
class Experiment(object):

    def __init__(self, reference:Network, target:Network, assignment_pair:AssignmentPair):
        self.reference = reference
        self.target = target
        self.assignment_pair = assignment_pair

    def __eq__(self, other)->bool:
        if not isinstance(other, Experiment):
            return False
        if not self.reference == other.reference:
            return False
        if not self.target == other.target:
            return False
        if not self.assignment_pair == other.assignment_pair:
            return False
        return True

    def __repr__(self):
        #####
        def networkRepr(network:Network):
            return f"(num_species: {network.num_species}, num_reaction: {network.num_reaction})"
        #####
        return f"reference: {networkRepr(self.reference)}, target: {networkRepr(self.target)}"
    
    def serialize(self):
        """Create a JSON string for the object.

        Returns:
            str: _description_
        """
        dict = {cn.S_ID: self.__class__.__name__,
                cn.S_REFERENCE: self.reference.serialize(), cn.S_TARGET: self.target.serialize(),
                cn.S_ASSIGNMENT_PAIR: self.assignment_pair.serialize()}
        return json.dumps(dict)
    
    @classmethod
    def deserialize(cls, json_str:str)->'Experiment':
        """Create an object from a JSON string.

        Args:
            json_str (str): _description_

        Returns:
            Experiment: _description_
        """
        dict = json.loads(json_str)
        reference = Network.deserialize(dict[cn.S_REFERENCE])
        target = Network.deserialize(dict[cn.S_TARGET])
        assignment_pair = AssignmentPair.deserialize(dict[cn.S_ASSIGNMENT_PAIR])
        return Experiment(reference=reference, target=target, assignment_pair=assignment_pair)
    

##############################
class ExperimentResult(object):

    def __init__(self, benchmark_runner:'BenchmarkRunner',
                 num_experiment:int=0, runtimes:List[float]=[], num_success:int=0,
          num_truncated:int=0, is_identicals:List[bool]=[], is_truncateds:List[bool]=[]):
        self.benchmark_runner = benchmark_runner
        self.num_experiment = num_experiment
        self.runtimes = runtimes
        self.num_success = num_success
        self.num_truncated = num_truncated
        self.is_identicals = is_identicals
        self.is_truncateds = is_truncateds

    def serialize(self)->str:
        """Create a JSON string for the object.

        Returns:
            str: _description_
        """
        dict = {cn.S_ID: self.__class__.__name__,
                S_BENCHMARK_RUNNER: self.benchmark_runner.serialize(),
                S_NUM_EXPERIMENT: self.num_experiment,
                S_RUNTIMES: self.runtimes, S_NUM_SUCCESS: self.num_success,
                S_NUM_TRUNCATED: self.num_truncated, S_IS_IDENTICALS: self.is_identicals,
                S_IS_TRUNCATEDS: self.is_truncateds}
        return json.dumps(dict)
    
    @classmethod
    def deserialize(cls, json_str:str)->'ExperimentResult':
        """Create an object from a JSON string.

        Args:
            json_str (str): _description_

        Returns:
            Experiment: _description_
        """
        dict = json.loads(json_str)
        benchmark_runner = BenchmarkRunner.deserialize(dict[S_BENCHMARK_RUNNER])
        num_experiment = dict[S_NUM_EXPERIMENT]
        runtimes = dict[S_RUNTIMES]
        num_success = dict[S_NUM_SUCCESS]
        num_truncated = dict[S_NUM_TRUNCATED]
        is_identicals = dict[S_IS_IDENTICALS]
        is_truncateds = dict[S_IS_TRUNCATEDS]
        return ExperimentResult(benchmark_runner, num_experiment=num_experiment,
                runtimes=runtimes, num_success=num_success, num_truncated=num_truncated,
                is_identicals=is_identicals, is_truncateds=is_truncateds)

    def __eq__(self, other)->bool:
        if not isinstance(other, ExperimentResult):
            return False
        if self.num_experiment != other.num_experiment:
            return False
        if not np.all(self.runtimes == other.runtimes):
            return False
        if self.num_success != other.num_success:
            return False
        if self.num_truncated != other.num_truncated:
            return False
        if not np.all(self.is_identicals == other.is_identicals):
            return False
        if not np.all(self.is_truncateds == other.is_truncateds):
            return False
        return True

    def __repr__(self):
        frac_truncated = self.num_truncated/self.num_experiment
        frac_success = self.num_success/self.num_experiment
        return f"num_experiment: {self.num_experiment}, frac_success: {frac_success}, " + \
                f"frac_truncated: {frac_truncated}, avg_runtime: {np.mean(self.runtimes)}"
    
    def getIdenticals(self)->List[Experiment]:
        """
        Gets the experiments that found an identical subnetwork.

        Returns:
            List[Experiment]
        """
        return [e for i, e in enumerate(self.benchmark_runner.experiments) if self.is_identicals[i]] 

    def getTruncateds(self)->List[Experiment]:
        """
        Gets the experiments whose search was truncated.

        Returns:
            List[Experiment]
        """
        return [e for i, e in enumerate(self.benchmark_runner.experiments) if self.is_truncateds[i]]


##############################
class BenchmarkRunner(object):

    def __init__(self, 
                num_experiment:int=10, 
                reference_size:int=3,
                expansion_factor:int=1,
                is_identical:bool=True,
                identity=cn.ID_WEAK):
        """
        Args:
            num_experiment (int): number of experiments in the benchmark
            reference_size (int)
            expansion_factor (int): Integer factor for size of target relative to reference
            is_identical (bool): If True, the target has a subnet equivalent to the reference.
            identity (str)
        """
        self.reference_size = reference_size
        self.expansion_factor = expansion_factor
        self.identity = identity
        self.num_experiment = num_experiment
        self.is_identical = is_identical
        self.experiments = [self.makeExperiment(is_identical=self.is_identical)]*num_experiment

    def __eq__(self, other)->bool:
        if not isinstance(other, BenchmarkRunner):
            return False
        if self.reference_size != other.reference_size:
            return False
        if self.expansion_factor != other.expansion_factor:
            return False
        if self.identity != other.identity:
            return False
        if self.num_experiment != other.num_experiment:
            return False
        if len(self.experiments) != len(other.experiments):
            return False
        for idx, experiment in enumerate(self.experiments):
            if not experiment == other.experiments[idx]:
                return False
        return True

    def makeExperiment(self, is_identical=True)->Experiment:
        """Construct the elements of an experiment.

        Args:
            is_identical (bool): If True, the target has a subnet equivalent to the reference.

        Returns:
            Experiment
        """
        #####
        def changeValue(value):
            if value == 0:
                return 1
            else:
                return 0
        #####
        reference = Network.makeRandomNetworkByReactionType(self.reference_size, is_prune_species=False)
        target_size = self.reference_size*self.expansion_factor
        filler_size = target_size - self.reference_size
        if filler_size == 0:
            # Do not expand the target
            target, assignment_pair = reference.permute()
        else:
            # Expand the target
            #   Create the left part of the expanded target
            left_filler_arr = np.zeros((filler_size, self.reference_size))
            xreactant_arr = np.vstack([reference.reactant_nmat.values, left_filler_arr])
            xproduct_arr = np.vstack([reference.product_nmat.values, left_filler_arr])
            #  Create the expanded arrays
            right_filler = Network.makeRandomNetworkByReactionType(num_reaction=filler_size,
                  num_species=target_size, is_prune_species=False)
            xreactant_arr = np.hstack([xreactant_arr, right_filler.reactant_nmat.values])
            xproduct_arr = np.hstack([xproduct_arr, right_filler.product_nmat.values])
            xtarget = Network(xreactant_arr, xproduct_arr)
            #  Merge the left and right parts
            target, assignment_pair = xtarget.permute()
            assignment_pair = assignment_pair.resize(self.reference_size)
        if not is_identical:
            i_species, i_reaction = np.random.randint(0, reference.num_species),  \
                  np.random.randint(0, reference.num_reaction)
            max_val = np.max(reference.reactant_nmat.values)
            # Insert an impossible value
            reference.reactant_nmat.values[i_species, i_reaction] =  max_val + 1
            reference.standard_nmat.values = reference.product_nmat.values - reference.reactant_nmat.values
        return Experiment(reference=reference, target=target, assignment_pair=assignment_pair)
    
    def run(self)->ExperimentResult:
        """Run the benchmark.

        Returns:
            ExperimentResult
        """
        runtimes = []
        num_success = 0
        num_truncated = 0
        is_identicals = []
        is_truncateds = []
        for experiment in self.experiments:
            start_time = time.process_time()
            result = experiment.reference.isStructurallyIdentical(experiment.target,
                  identity=self.identity, is_subnet=True)
            runtimes.append(time.process_time() - start_time)
            is_identical = any([ap == experiment.assignment_pair for ap in result.assignment_pairs])
            is_identicals.append(is_identical)
            is_truncateds.append(result.is_truncated)
            if is_identical:
                num_success += 1
            if result.is_truncated:
                num_truncated += 1
        return ExperimentResult(self, num_experiment=self.num_experiment, runtimes=runtimes,
              num_success=num_success, num_truncated=num_truncated,
              is_identicals=is_identicals, is_truncateds=is_truncateds)
    
    def exportExperimentsAsCSV(self, directory:str)->None:
        """Creates two subdirectores, reference and target, with the experiments in CSV format.

        Args:
            directory (str): _description_
        """
        #####
        def mkSubdir(subdir:str):
            subdir = os.path.join(directory, subdir)
            if os.path.exists(subdir):
                shutil.rmtree(subdir)
            os.makedirs(subdir)
            return subdir
        #####
        def writeSubdir(is_reference:bool=True):
            if is_reference:
                subdir = mkSubdir(REFERENCE)
            else:
                subdir = mkSubdir(TARGET)
            #
            for idx, experiment in enumerate(self.experiments):
                if is_reference:
                    network = experiment.reference
                else:
                    network = experiment.target
                network_str = network.makeCSVNetwork(identity=self.identity)
                path = os.path.join(subdir, f"{idx}.txt")
                with open(path, "w") as file:
                    file.write(network_str)
        #####
        # Create the CSV files
        writeSubdir(is_reference=True)
        writeSubdir(is_reference=False)

    def serialize(self)->str:
        """Create a JSON string for the object.

        Returns:
            str: _description_
        """
        experiments = [experiment.serialize() for experiment in self.experiments]
        dict = {cn.S_ID: self.__class__.__name__,
                S_REFERENCE_SIZE: self.reference_size, S_EXPANSION_FACTOR: self.expansion_factor,
                S_NUM_EXPERIMENT: self.num_experiment, S_IS_IDENTICAL: self.is_identical,
                S_EXPERIMENTS: experiments, cn.S_IDENTITY: self.identity}
        return json.dumps(dict)
    
    @classmethod
    def deserialize(cls, serialization_str:str)->'BenchmarkRunner':
        """Create an object from a JSON string.

        Args:
            serialization_str (str): _description_

        Returns:
            BenchmarkRunner: _description_
        """
        dict = json.loads(serialization_str)
        reference_size = dict[S_REFERENCE_SIZE]
        expansion_factor = dict[S_EXPANSION_FACTOR]
        num_experiment = dict[S_NUM_EXPERIMENT]
        is_identical = dict[S_IS_IDENTICAL]
        identity = dict[cn.S_IDENTITY]
        experiments = [Experiment.deserialize(exp_str) for exp_str in dict[S_EXPERIMENTS]]
        benchmark_runner = BenchmarkRunner(num_experiment=num_experiment, reference_size=reference_size,
              expansion_factor=expansion_factor, is_identical=is_identical, identity=identity)
        benchmark_runner.experiments = experiments
        return benchmark_runner