from .db import DB
from dreamer.utils.schemes.db_scheme import DBModScheme
from dreamer.utils.constants.constant import Constant
from ...errors import MissingPath
from . import config as v1_config

from dreamer.configs import (
    db_config,
    sys_config,
    DBUsages
)
from dreamer.utils.types import *
from dreamer.utils.schemes.module import CatchErrorInModule


class BasicDBMod(DBModScheme):
    """
    A module for maintaining the inspiration function database.
    """

    def __init__(self, path: Optional[str] = v1_config.DEFAULT_PATH, json_path: Optional[str] = None):
        """
        :param path: Path to the database file.
        :param json_path: If DB usage uses a JSON file, a path to the JSON file is expected.
        """
        super().__init__(
            description='Database module for inspiration function management',
            version='1'
        )
        self.db = DB(path)
        self.json_path = json_path

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self, constants: Optional[List[Constant] | Constant] = None) -> Dict[Constant, List[ShiftCMF]] | None:
        def classify_usage(usage: DBUsages) -> Optional[dict]:
            match usage:
                case DBUsages.RETRIEVE_DATA:
                    return {constant: self.db.select(constant) for constant in constants}
                case DBUsages.STORE_DATA:
                    try:
                        if not self.json_path:
                            raise MissingPath(MissingPath.default_msg)
                        self.db.from_json(self.json_path)
                    except Exception as e:
                        raise e
                case _:
                    raise ValueError(f"Invalid usage: {usage.name} ")

        if not (usage := db_config.USAGE) in v1_config.ALLOWED_USAGES:
            raise ValueError(f"Invalid usage: {usage.name} "
                             f"usages are: {[usage.name for usage in v1_config.ALLOWED_USAGES]}")
        if constants is None:
            constants = sys_config.CONSTANTS
        elif isinstance(constants, Constant):
            constants = [constants]
        match usage:
            case DBUsages.STORE_THEN_RETRIEVE:
                classify_usage(DBUsages.STORE_DATA)
                return classify_usage(DBUsages.RETRIEVE_DATA)
            case DBUsages.RETRIEVE_DATA:
                return classify_usage(usage)
            case _:
                raise NotImplementedError(f"Invalid usage: {usage.name} ")
