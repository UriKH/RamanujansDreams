from dataclasses import dataclass, fields

from .configurable import Configurable


@dataclass
class LogConfig(Configurable):
    SLEEP_TO_PRINT: bool = False
    PROFILE: bool = False


logging_config: LogConfig = LogConfig()
