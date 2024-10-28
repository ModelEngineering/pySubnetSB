from sirn.benchmark_runner import BenchmarkRunner, Experiment, ExperimentResult # type: ignore
from sirn.network import Network  # type: ignore
import sirn.constants as cn  # type: ignore

import os
import shutil
import unittest


IGNORE_TEST = False
IS_PLOT = False
SIZE = 3
EXPANSION_FACTOR = 2
IDENTITY = cn.ID_WEAK
REFERENCE_DIR = os.path.join(cn.TEST_DIR, "reference")
TARGET_DIR = os.path.join(cn.TEST_DIR, "target")
REMOVE_DIRS = [REFERENCE_DIR, TARGET_DIR]


#############################
# Tests
#############################
class TestBenchmarkRunner(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.benchmark_runner = BenchmarkRunner(reference_size=SIZE, expansion_factor=EXPANSION_FACTOR,
              identity=IDENTITY)
        
    def tearDown(self):
        self.remove()
    
    def remove(self):
        for dir in REMOVE_DIRS:
            if os.path.exists(dir):
                shutil.rmtree(dir)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.benchmark_runner.reference_size, SIZE)
        self.assertEqual(self.benchmark_runner.expansion_factor, EXPANSION_FACTOR)
        self.assertEqual(self.benchmark_runner.identity, IDENTITY)
        self.assertEqual(len(self.benchmark_runner.experiments), self.benchmark_runner.num_experiment)

    def testMakeStructurallySimilarExperiment(self):
        if IGNORE_TEST:
            return
        size = 3
        num_iteration = 10
        for expansion_factor in [2, 1, 3]:
            target_size = size*expansion_factor
            for _ in range(num_iteration):
                benchmark_runner = BenchmarkRunner(reference_size=size, expansion_factor=expansion_factor,
                    identity=IDENTITY)
                experiment = benchmark_runner.makeExperiment()
                self.assertEqual(experiment.reference.num_species, SIZE)
                self.assertEqual(experiment.reference.num_reaction, SIZE)
                self.assertEqual(experiment.target.num_species, target_size)
                self.assertEqual(len(experiment.assignment_pair.species_assignment), size)
                self.assertEqual(len(experiment.assignment_pair.reaction_assignment), size)
                self.assertEqual(experiment.target.num_reaction, target_size)
                self.assertTrue(experiment.target.permute(
                      assignment_pair=experiment.assignment_pair)[0].isEquivalent(experiment.reference))
    
    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        serialization_str = self.benchmark_runner.serialize()
        benchmark_runner = BenchmarkRunner.deserialize(serialization_str)
        self.assertEqual(self.benchmark_runner, benchmark_runner)

    def testRun(self):
        if IGNORE_TEST:
            return
        num_experiment = 3
        reference_size = 5
        for identity in cn.ID_LST:
            for expansion_factor in [2, 10]:
                benchmark_runner = BenchmarkRunner(reference_size=reference_size,
                    num_experiment=num_experiment,
                    expansion_factor=expansion_factor,
                    is_identical=True,
                    identity=identity)
                experiment_result = benchmark_runner.run()
                self.assertEqual(len(experiment_result.runtimes), benchmark_runner.num_experiment)
                count = experiment_result.num_success + experiment_result.num_truncated
                count = min(count, num_experiment)
                self.assertEqual(num_experiment, count)

    def testRunNotIdentical(self):
        if IGNORE_TEST:
            return
        num_experiment = 3
        reference_size = 5
        for identity in [cn.ID_WEAK]:
            benchmark_runner = BenchmarkRunner(reference_size=reference_size,
                num_experiment=num_experiment,
                expansion_factor=1,
                is_identical=False,
                identity=identity)
            experiment_result = benchmark_runner.run()
            self.assertEqual(len(experiment_result.runtimes), benchmark_runner.num_experiment)
            self.assertEqual(experiment_result.num_success, 0)

    def testExportExperimentAsCSV(self):
        if IGNORE_TEST:
            return
        benchmark_runner = BenchmarkRunner(reference_size=10, expansion_factor=10,
              identity=IDENTITY, num_experiment=10, is_identical=False)
        benchmark_runner.exportExperimentsAsCSV(cn.TEST_DIR)
        reference_files = os.listdir(REFERENCE_DIR)
        target_files = os.listdir(TARGET_DIR)
        self.assertEqual(len(reference_files), len(target_files))


##################################################
class TestExperiment(unittest.TestCase):

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        reference = Network.makeRandomNetworkByReactionType(SIZE, is_prune_species=False)
        target, assignment_pair = reference.permute()
        experiment = Experiment(reference=reference, target=target, assignment_pair=assignment_pair)
        serialization_str = experiment.serialize()
        new_experiment = Experiment.deserialize(serialization_str)
        self.assertEqual(experiment, new_experiment)


##################################################
class TestExperimentResult(unittest.TestCase):

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        benchmark_runner = BenchmarkRunner(reference_size=SIZE, expansion_factor=EXPANSION_FACTOR,
              identity=IDENTITY)
        experiment_result = benchmark_runner.run()
        serialization_str = experiment_result.serialize()
        new_experiment_result = ExperimentResult.deserialize(serialization_str)
        self.assertEqual(experiment_result, new_experiment_result)


if __name__ == '__main__':
    unittest.main()