import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TEST_DIR = os.path.join(PROJECT_DIR, 'tests')
OSCILLATOR_DIRS = [
        "Oscillators_May_28_2024_8898",
        "Oscillators_June_9_2024_18070",
        "Oscillators_June_10_A_11515",
        "Oscillators_June_10_B_10507",
        "Oscillators_June_11_10160",
        "Oscillators_June_11_A_2024_6877",
        "Oscillators_June_11_B_2024_7809",
        "Oscillators_DOE_JUNE_10_17565",
        "Oscillators_DOE_JUNE_12_A_30917",
        "Oscillators_DOE_JUNE_12_B_41373",
        "Oscillators_DOE_JUNE_12_C_27662",
        ]