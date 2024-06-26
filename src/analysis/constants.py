import sirn.constants as cnn
import os

OSCILLATOR_PROJECT_DIR = "/Users/jlheller/home/Technical/repos/OscillatorDatabase"
IDENTITY_STRONG_PATH_DCT = {d: os.path.join(cnn.DATA_DIR, "strong_identity_1B", "identity_" + d + ".txt") for d in cnn.OSCILLATOR_DIRS}
IDENTITY_WEAK_PATH_DCT = {d: os.path.join(cnn.DATA_DIR, "weak_identity_1B", "identity_" + d + ".txt") for d in cnn.OSCILLATOR_DIRS}