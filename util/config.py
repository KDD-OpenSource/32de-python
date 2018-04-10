from os import path
from logging.config import dictConfig

# algorithms
RESEARCH_MODE = "research"
BASELINE_MODE = "baseline"
# development
RANDOM_STATE = 42  # change it to something random if it should not be reconstructive.
# server
REACT_PORT = 3000
API_PORT = 8000
SERVER_PATH = 'localhost'
# Data sets
RATED_DATASETS_PATH = path.join('rated_datasets')
MOCK_DATASETS_DIR = path.join('tests', 'data')
# Configuration for sessions saved on the file system
SESSION_CACHE_DIR = path.join('tmp', 'sessions')
SESSION_THRESHOLD = 500
SESSION_MODE = '0700'


def set_up_logger():
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s from %(name)s: %(message)s',
        }},
        'handlers': {
            'default': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': 'DEBUG'
            }},
        'loggers': {
            'MetaExp': {
                'handlers': ['default']
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': []
        },
    })
