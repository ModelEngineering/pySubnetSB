import sirn.constants as cnn # type: ignore
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
TEST_DIR = os.path.join(PROJECT_DIR, 'analysis_tests')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

OSCILLATOR_PROJECT_DIR = "/Users/jlheller/home/Technical/repos/OscillatorDatabase"