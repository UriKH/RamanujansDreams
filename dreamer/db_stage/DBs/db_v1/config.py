"""
Module configuration - specific to DB db_v1
"""
from dreamer.configs.database import DBUsages

DEFAULT_PATH = './families_v1.db'
ALLOWED_USAGES = [                  # allowed usages of the DB
    DBUsages.RETRIEVE_DATA,
    DBUsages.STORE_DATA,
    DBUsages.STORE_THEN_RETRIEVE
]
MULTIPLE_CONSTANTS = False          # allow operations on multiple constants
# PARALLEL = False                    # Allow parallelism
