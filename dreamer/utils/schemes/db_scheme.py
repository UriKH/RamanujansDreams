import os.path
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict
from tqdm import tqdm
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.module import Module, CatchErrorInModule
from dreamer.utils.types import *
from dreamer.loading.funcs.formatter import Formatter
from dreamer.loading.config import *
from dreamer.configs import sys_config
import json


class DBModScheme(Module):
    @classmethod
    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def aggregate(cls, dbs: List["DBModScheme"],
                  constants: Optional[List[Constant] | Constant] = None,
                  close_after_exec: bool = False) -> DefaultDict[Constant, Set[ShiftCMF]]:
        """
        Aggregate results from multiple DBModConnector instances.
        i.e., combine data from multiple databases
        :param dbs: A list of database instances given by System
        :param constants: A list of constants to search for. If None, search for constants defined in the configuration
                            file in 'configs.database.py'.
        :param close_after_exec: Close the database after execution
        :return: A dictionary mapping each constant to a list of CMFs and their respective shifts
        """
        results = defaultdict(set)
        for db in tqdm(dbs, desc=f'Extracting data from DBs', **sys_config.TQDM_CONFIG):
            if not issubclass(db.__class__, cls):
                raise ValueError(f"Invalid DBModConnector instance: {db}")
            for const, l in db.execute(constants).items():
                results[const] = set(results.get(const, []) + l)
            if close_after_exec:
                del db
        return results

    @abstractmethod
    def execute(self, constants: Optional[List[Constant] | Constant] = None) -> Dict[Constant, List[ShiftCMF]] | None:
        raise NotImplementedError

    @staticmethod
    def export_future_append_to_json(
            functions: Optional[Union['Formatter', List['Formatter']]] = None,
            path: Optional[str] = None,
            exits_ok: bool = False
    ) -> Dict[str, Any]:
        """
        Export a future command into json
        :param functions:
        :param path:
        :return:
        """
        path = path if path else 'command'
        path = path if path.endswith('.json') else path + '.json'

        if os.path.exists(path) and not exits_ok:
            raise FileExistsError(f"File {path} already exists")

        if isinstance(functions, Formatter):
            functions = [functions]
        jsons = []
        for func in functions:
            jsons.append(func.to_json_obj())
        jsons = {COMMAND_ANNOTATE: DBScheme.append.__name__, DATA_ANNOTATE: jsons}
        with open(path, 'w') as f:
            json.dump(jsons, f, indent=4)
        return jsons


class DBScheme(ABC):
    @abstractmethod
    def select(self, constant: Constant) -> List[ShiftCMF]:
        """
        Retrieve the CMFs of the inspiration funcs corresponding to the given constant.
        :param constant: The constant for which to retrieve the CMFs.
        :return: A list of tuples (CMF, shifts) for each inspiration function.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, constant: Constant, funcs: List[Formatter] | Formatter, replace: bool = False) -> None:
        """
        Set the inspiration funcs corresponding to the given constant.
        :param constant: The constant for which to retrieve the CMFs.
        :param funcs: The collection of inspiration-funcs.
        :param replace: If true, replace the existing inspiration funcs.
        :raises ConstantAlreadyExists: If the constant already exists and replace is false.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, constant: Constant | List[Constant], funcs: Optional[List[Formatter] | Formatter] = None,
               delete_const: bool = False) -> List[Constant] | None:
        """
        Remove all the funcs from all the constants provided.
        :param constant: A constant or a list of constants to remove from.
        :param funcs: A function or a list of funcs to remove from the constants. If None, remove the constant.
        :param delete_const: If True, delete the constant from the database if funcs is None.
         Otherwise, just remove all its funcs
        :return: A list of constants that were removed.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all the funcs from all the constants.
        """
        raise NotImplementedError

    @abstractmethod
    def from_json(self, path: str) -> None:
        """
        Execute commands via JSON. View the format in the JSONError class.
        :param path: Path to the JSON file.
        """
        raise NotImplementedError

    @abstractmethod
    def replace(self, constant: Constant, funcs: List[Formatter] | Formatter) -> None:
        raise NotImplementedError

    @abstractmethod
    def append(self, constant: Constant, funcs: List[Formatter] | Formatter) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, constant: Constant, funcs: List[Formatter] | Formatter) -> None:
        raise NotImplementedError
