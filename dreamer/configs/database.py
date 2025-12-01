"""
Global config file for system flow regarding DBs
"""
from dataclasses import dataclass
from enum import Enum, auto

from .configurable import Configurable


class DBUsages(Enum):
    RETRIEVE_DATA = auto()
    STORE_DATA = auto()
    STORE_THEN_RETRIEVE = auto()


@dataclass
class DBConfig(Configurable):
    USAGE: DBUsages = DBUsages.STORE_THEN_RETRIEVE      # execute DBs retrieve if retrieve is an option


db_config: DBConfig = DBConfig()
