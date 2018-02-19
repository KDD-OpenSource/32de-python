import os

# algorithms
RESEARCH_MODE = "research"
BASELINE_MODE = "baseline"
# server
REACT_PORT = 3000
API_PORT = 8000
# Data sets
# TODO review this
# all these paths are relative to the server.py script
# your os.getcwd() should be "api/"
ROTTEN_TOMATO_PATH = os.path.join('..', 'tests', 'data', 'rotten_tomatoes')
RATED_DATASETS_PATH = os.path.join('..', 'rated_datasets')
MOCK_DATASETS_DIR = os.path.join('..', 'tests', 'data')
# Configuration for sessions saved on the file system
SESSION_CACHE_DIR = os.path.join('..', 'tmp', 'sessions')
SESSION_THRESHOLD = 500
SESSION_MODE = '0700'