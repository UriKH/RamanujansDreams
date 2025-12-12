from .system import sys_config
from .database import db_config, DBUsages
from .analysis import analysis_config
from .search import search_config
from .extraction import extraction_config
from ..utils.logger import Logger
from typing import Dict, List


class ConfigManager:
    """
    Global configuration manager for the search system
    """
    system = sys_config
    database = db_config
    extraction = extraction_config
    analysis = analysis_config
    search = search_config

    def configure(self, **overrides):
        """
        Override multiple configs at once.

        Example:
            Config.configure(system={"CONSTANT": "X"}, database={"DB_NAME": "foo"})
        """
        warned = False
        for section, values in overrides.items():
            cfg = getattr(self, section, None)
            if cfg is not None:
                for key, val in values.items():
                    setattr(cfg, key, val)
            else:
                Logger(
                    f'section {section} is not defined, try: system / database / analysis / search',
                    Logger.Levels.inform
                ).log()
                if not warned:
                    warned = True
                    Logger(f'Note that these are the builtin attributes: '
                           f'{system}\n\n{database}\n\n{extraction}\n\n{analysis}\n\n{search}', Logger.Levels.inform
                           ).log()

    def get_configurables(self) -> Dict[str, List[str]]:
        return {
            'system': self.system.get_configurations(),
            'database': self.database.get_configurations(),
            'extraction': self.extraction.get_configurations(),
            'analysis': self.analysis.get_configurations(),
            'search': self.search.get_configurations()
        }


config = ConfigManager()

__all__ = [
    'config',
    'sys_config',
    'db_config',
    'extraction_config',
    'DBUsages',
    'analysis_config',
    'search_config'
]
