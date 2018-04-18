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
# Redis Configuration
REDIS_PORT = 6379
REDIS_HOST = '172.16.19.193'
REDIS_PASSWORD = None

MAX_META_PATH_LENGTH = 6

AVAILABLE_DATA_SETS = [
    {
        'name': 'Freebase',
        'url': 'https://hpi.de/mueller/metaexp-demo-neo4j',
        'bolt-url': 'bolt://172.20.14.22:7717',
        'username': 'neo4j',
        'password': 'neo4j'
    },
    {
        'name': 'Helmholtz',
        'url': 'https://hpi.de/mueller/metaexp-demo-neo4j-2',
        'bolt-url': 'bolt://172.20.14.22:7697',
        'username': 'neo4j',
        'password': 'neo4j'
    }
]

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
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'filename': 'log/debug.log',
                'mode': 'w',
                'level': 'DEBUG'
            },
        },
        'loggers': {
            'MetaExp': {
                'handlers': ['default', 'file']
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': []
        },
    })
