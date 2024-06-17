import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TEST_DIR = os.path.join(PROJECT_DIR, 'tests')