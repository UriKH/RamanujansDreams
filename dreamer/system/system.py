from collections import defaultdict
import networkx as nx
from itertools import combinations
from enum import Enum, auto
import os

from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.schemes.db_scheme import DBModScheme
from dreamer.db_stage.funcs.formatter import Formatter
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from ..utils.storage import Exporter, Importer, Formats
from ..utils.types import *
from ..utils.logger import Logger
from ..utils.constant_transform import *


class System:
    """
    A class representing the System itself.
    """

    class RunModes(Enum):
        DB_ONLY = auto()
        JSON_TO_DB = auto()
        JSON_ONLY = auto()
        MANUAL = auto()

    def __init__(self,
                 if_srcs: List[DBModScheme | str | Formatter],
                 analyzers: List[Type[AnalyzerModScheme] | str | Searchable],
                 searcher: Type[SearcherModScheme]):
        """
        Constructing a system runnable instance for a given combination of modules.
        :param if_srcs: A list of DBModScheme instances used as sources
        :param analyzers: A list of AnalyzerModScheme types used for prioritization + preparation before the search
        :param searcher: A SearcherModScheme type used to deepen the search done by the analyzers
        """
        self.if_srcs = if_srcs
        self.analyzers = analyzers
        self.searcher = searcher
        # self.if_srcs_contain_dbs = any(isin)

    def run(self, constants: List[str] | str = None):
        """
        Run the system given the constants to search for.
        :param constants: if None, search for constants defined in the configuration file in 'configs.database.py'.
        :return:
        """
        if not constants:
            constants = sys_config.CONSTANTS
        elif isinstance(constants, str):
            constants = [constants]

        constants = get_constants(constants)
        cmf_data = self.__db_stage(constants)
        if path := sys_config.EXPORT_CMFS:
            os.makedirs(path, exist_ok=True)

            for const, l in cmf_data.items():
                # Sanitize filename (optional, avoids invalid characters)
                safe_key = "".join(c for c in const if c.isalnum() or c in ('-', '_'))
                path = os.path.join(path, safe_key)

                Exporter.export(path, exists_ok=True, clean_exists=True, data=l, fmt=Formats.PICKLE)
                Logger(
                    f'CMFs for {const} exported to {path}', Logger.Levels.info
                ).log(msg_prefix='\n')

        for constant, funcs in cmf_data.items():
            functions = '\n'
            for i, func in enumerate(funcs):
                functions += f'{i+1}. {func}\n'
            Logger(
                f'Searching for {constant} using inspiration functions: {functions}', Logger.Levels.info
            ).log(msg_prefix='\n')

        priorities = self.__analysis_stage(cmf_data)
        if path := sys_config.EXPORT_ANALYSIS_PRIORITIES:
            os.makedirs(path, exist_ok=True)

            for const, l in priorities.items():
                # Sanitize filename (optional, avoids invalid characters)
                safe_key = "".join(c for c in const if c.isalnum() or c in ('-', '_'))
                path = os.path.join(path, safe_key)

                Exporter.export(path, exists_ok=True, clean_exists=True, data=l, fmt=Formats.PICKLE)
                Logger(
                    f'Priorities for {const} exported to {path}', Logger.Levels.info
                ).log(msg_prefix='\n')

        self.__search_stage(priorities)

    def __db_stage(self, constants: Dict[str, Any]) -> Dict[str, List[ShiftCMF]]:
        modules = []
        cmf_data = defaultdict(set)

        for db in self.if_srcs:
            if isinstance(db, DBModScheme):
                modules.append(db)
            elif isinstance(db, str):
                f_data = Importer.imprt(db)
                for obj in f_data:
                    cmf_data[obj.const].add(obj.to_cmf())
            elif isinstance(db, Formatter):
                cmf_data[db.const].add(db.to_cmf())
            else:
                raise ValueError(f'string is not a json file: {db}')

        cmf_data_2 = DBModScheme.aggregate(modules, list(constants.keys()), True)

        for const in cmf_data_2.keys():
            cmf_data[const].union(cmf_data_2[const])

        # convert back to dict[str, list]
        as_list = dict()
        for k, v in cmf_data.items():
            if k not in constants:
                Logger(
                    f'constant {k} is not in search list, its inspiration functions will be ignored',\
                    level=Logger.Levels.warning
                ).log(msg_prefix='\n')
                continue
            as_list[k] = list(v)
        return as_list

    def __analysis_stage(self, cmf_data: Dict[str, List[ShiftCMF]]) -> Dict[str, List[Searchable]]:
        """
        Preform the analysis stage work
        :param cmf_data: data produced in the DB stage
        :return: The results of the analysis
        """
        analyzers = []
        results = defaultdict(set)

        for analyzer in self.analyzers:
            match analyzer:
                case t if issubclass(t, AnalyzerModScheme):
                    analyzers.append(analyzer)
                case Searchable():
                    results[analyzer.const_name].add(analyzer)
                case str():
                    f_data = Importer.imprt(analyzer)
                    for obj in f_data:
                        results[obj.const_name].add(obj)
                case _:
                    raise TypeError(f'unknown analyzer type {analyzer}')

        analyzers_results = [analyzer(cmf_data).execute() for analyzer in analyzers]
        priorities = self.__aggregate_analyzers(analyzers_results)

        # add unprioritized elements to the end
        for c, l in results.items():
            if c not in priorities:
                continue
            diff = results[c].difference(set(cmf_data[c]))
            priorities[c].extend(diff)
        return priorities

    def __search_stage(self, priorities: dict[str, List[Searchable]]):
        # results = dict()
        for data in priorities.values():
            self.searcher(data, True).execute()
            # results[const] = s.execute()

            # if path := sys_config.EXPORT_SEARCH_RESULTS:
            #     os.makedirs(path, exist_ok=True)
                # TODO: print result summary for this constant, real export in the searcher class

        for const in priorities.keys():
            best_delta = -sp.oo
            best_sv = None
            best_space = None
            dir_path = os.path.join(sys_config.EXPORT_SEARCH_RESULTS, const)

            stream_gen = Importer.import_stream(dir_path)
            for dm in stream_gen:
                delta, sv = dm.best_delta
                if delta is None:
                    continue
                if best_delta < delta:
                    best_delta, best_sv = delta, sv
            Logger(
                f'Best delta for "{const}": {best_delta} in trajectory: {best_sv}',
                Logger.Levels.info
            ).log(msg_prefix='\n')

    @staticmethod
    def validate_constant(constant: str, throw: bool = False) -> bool:
        """
        Check if a constant is defined in sympy.
        :param constant: Constant name as string
        :param throw: if True, throw an error on fail
        :raise UnknownConstant if constant is unknown
        :return: True if constant is defined in sympy.
        """
        try:
            get_const_as_sp(constant)
            return True
        except UnknownConstant as e:
            if throw:
                raise e
            return False

    @staticmethod
    def __aggregate_analyzers(dicts: List[Dict[str, List[Searchable]]]) -> Dict[str, List[Searchable]]:
        """
        Aggregates the priority lists from several analyzers into one
        :param dicts: A list of mappings from constant name to a list of its relevant subspaces
        :return: The aggregated priority dictionaries
        """
        all_keys = set().union(*dicts)
        result = {}

        for key in all_keys:
            lists = [d[key] for d in dicts if key in d]
            prefs = defaultdict(int)
            searchables = set().union(*lists)

            # Count preferences
            for lst in lists:
                for i, a in enumerate(lst[:-1]):
                    for j, b in enumerate(lst[i + 1:]):
                        prefs[(a, b)] += 1  #(j - i) * 1. / len(lst)

            G = nx.DiGraph()
            G.add_nodes_from(searchables)
            for a, b in combinations(searchables, 2):
                if prefs[(a, b)] > prefs[(b, a)]:
                    G.add_edge(a, b)
                elif prefs[(a, b)] < prefs[(b, a)]:
                    G.add_edge(b, a)
                else:
                    if hash(a) > hash(b):
                        G.add_edge(a, b)
                    else:
                        G.add_edge(b, a)

            try:
                consensus = list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                raise Exception('This was not supposed to happen')
            result[key] = consensus
        return result
