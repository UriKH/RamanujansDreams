from collections import defaultdict
import networkx as nx
from itertools import combinations
from enum import Enum, auto
import os

from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.schemes.db_scheme import DBModScheme
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.extraction_scheme import ExtractionModScheme
from ..utils.storage import Exporter, Importer, Formats
from ..utils.types import *
from ..utils.logger import Logger
from ..utils.constants.constant import Constant
from dreamer.configs.system import sys_config
from dreamer.configs.extraction import extraction_config
from functools import partial


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
                 extractor: Optional[Type[ExtractionModScheme]],
                 analyzers: List[Type[AnalyzerModScheme] | str | Searchable],
                 searcher: Type[SearcherModScheme] | partial[SearcherModScheme]):
        """
        Constructing a system runnable instance for a given combination of modules.
        :param if_srcs: A list of DBModScheme instances used as sources
        :param analyzers: A list of AnalyzerModScheme types used for prioritization + preparation before the search
        :param searcher: A SearcherModScheme type used to deepen the search done by the analyzers
        """
        self.if_srcs = if_srcs
        self.extractor = extractor
        self.analyzers = analyzers
        self.searcher = searcher

    def run(self, constants: List[str] | str | Constant | List[Constant] = None):
        """
        Run the system given the constants to search for.
        :param constants: if None, search for constants defined in the configuration file in 'configs.database.py'.
        :return:
        """
        if not constants:
            constants = sys_config.CONSTANTS

        if isinstance(constants, str | Constant):
            constants = [constants]

        as_obj = []
        for c in constants:
            if isinstance(c, str):
                if not Constant.is_registered(c):
                    raise ValueError(f'Constant "{c}" is not a registered constant.')
                else:
                    as_obj.append(Constant.get_constant(c))
            else:
                as_obj.append(c)

        constants = as_obj
        cmf_data = self.__loading_stage(constants)
        if path := sys_config.EXPORT_CMFS:
            os.makedirs(path, exist_ok=True)

            for const, l in cmf_data.items():
                # Sanitize filename (optional, avoids invalid characters)
                safe_key = "".join(c for c in const.name if c.isalnum() or c in ('-', '_'))
                const_path = os.path.join(path, safe_key)

                l = [ShiftCMF(scmf.cmf, scmf.shift, True) for scmf in l]

                Exporter.export(const_path, exists_ok=True, clean_exists=True, data=l, fmt=Formats.PICKLE)
                Logger(
                    f'CMFs for {const.name} exported to {const_path}', Logger.Levels.info
                ).log(msg_prefix='\n')

        for constant, funcs in cmf_data.items():
            functions = '\n'
            for i, func in enumerate(funcs):
                if not func.raw:
                    functions += f'{i+1}. {func}\n'
                else:
                    functions += f'{i+1}. Manually added CMF of dimension={func.cmf.dim()} and shift={func.shift}\n'
            Logger(
                f'Searching for {constant.name} using inspiration functions: {functions}', Logger.Levels.info
            ).log(msg_prefix='\n')

        # Extractor computes shards and saves the results in files in a format.
        # If extractor not provided, analysis will try to read from the default searchables directory
        shard_dict = None
        if self.extractor:
            shard_dict = self.extractor(cmf_data).execute()

        # analysis stage executes the analyzers. These will read the files
        priorities = self.__analysis_stage(shard_dict)
        filtered_priorities = dict()
        if path := sys_config.EXPORT_ANALYSIS_PRIORITIES:
            os.makedirs(path, exist_ok=True)

            for const, l in priorities.items():
                if not l:
                    Logger(
                        f'No shards remained after analysis. Run for constant "{const.name}" is stopped.',
                        Logger.Levels.warning
                    ).log(msg_prefix='\n')
                    continue

                # Sanitize filename (optional, avoids invalid characters)
                safe_key = "".join(c for c in const.name if c.isalnum() or c in ('-', '_'))
                const_path = os.path.join(path, safe_key)

                Exporter.export(const_path, exists_ok=True, clean_exists=False, data=l, fmt=Formats.PICKLE)
                Logger(
                    f'Priorities for {const.name} exported to {const_path}', Logger.Levels.info
                ).log(msg_prefix='\n')
                filtered_priorities[const] = l

        self.__search_stage(filtered_priorities)

    def __loading_stage(self, constants: List[Constant]) -> Dict[Constant, List[ShiftCMF]]:
        Logger('Loading CMFs ...', Logger.Levels.info).log(msg_prefix='\n')
        modules = []
        cmf_data = defaultdict(set)

        for db in self.if_srcs:
            if isinstance(db, DBModScheme):
                modules.append(db)
            elif isinstance(db, str):
                shift_cmf = Importer.imprt(db)
                cmf_data[Constant.get_constant(db.split('/')[-2])].add(shift_cmf)
            elif isinstance(db, Formatter):
                cmf_data[Constant.get_constant(db.const)].add(db.to_cmf())
            else:
                raise ValueError(f'string is not a json file: {db}')

        cmf_data_2 = dict()
        if modules:
            cmf_data_2 = DBModScheme.aggregate(modules, constants, True)

        for const in cmf_data_2.keys():
            cmf_data[const].update(cmf_data_2[const])

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

    def __analysis_stage(self, cmf_data: Optional[Dict[Constant, List[Searchable]]] = None) -> Dict[Constant, List[Searchable]]:
        """
        Preform the analysis stage work
        :param cmf_data: data produced in the DB stage
        :return: The results of the analysis
        """
        analyzers: List[Type[AnalyzerModScheme]] = []
        results = defaultdict(set)

        for analyzer in self.analyzers:
            match analyzer:
                case t if issubclass(t, AnalyzerModScheme):
                    analyzers.append(analyzer)
                case Searchable():
                    results[analyzer.const].add(analyzer)
                case str():
                    f_data = Importer.imprt(analyzer)
                    for obj in f_data:
                        results[obj].add(obj)
                case _:
                    raise TypeError(f'unknown analyzer type {analyzer}')

        if not cmf_data:
            cmf_data = {}
            for const_name in os.listdir(extraction_config.PATH_TO_SEARCHABLES):
                imp_stream = Importer.import_stream(f'{extraction_config.PATH_TO_SEARCHABLES}\\{const_name}')
                const_shards = []
                for shards in imp_stream:
                    const_shards += shards
                if const_shards:
                    cmf_data[const_shards[0].const] = const_shards

        analyzers_results = [analyzer(cmf_data).execute() for analyzer in analyzers]
        priorities = self.__aggregate_analyzers(analyzers_results)

        # add unprioritized elements to the end
        for c, l in results.items():
            if c not in priorities:
                continue
            diff = results[c].difference(set(cmf_data[c]))
            priorities[c].extend(diff)
        return priorities

    def __search_stage(self, priorities: Dict[Constant, List[Searchable]]):
        # results = dict()
        for data in priorities.values():
            self.searcher(data, sys_config.USE_LIReC).execute()
            # results[const] = s.execute()

            # if path := sys_config.EXPORT_SEARCH_RESULTS:
            #     os.makedirs(path, exist_ok=True)
                # TODO: print result summary for this constant, real export in the searcher class

        for const in priorities.keys():
            best_delta = -sp.oo
            best_sv = None
            best_space = None
            dir_path = os.path.join(sys_config.EXPORT_SEARCH_RESULTS, const.name)

            stream_gen = Importer.import_stream(dir_path)
            for dm in stream_gen:
                delta, sv = dm.best_delta
                if delta is None:
                    continue
                if best_delta < delta:
                    best_delta, best_sv = delta, sv

            if best_sv is None:
                # Should not happen
                Logger('No best delta found').log(msg_prefix='\n')
            else:
                Logger(
                    f'Best delta for "{const.name}" found by the searcher is {best_delta}\n'
                    f'* Trajectory: {best_sv.trajectory} \n* Start: {best_sv.start}',
                    Logger.Levels.info
                ).log(msg_prefix='\n')

    @staticmethod
    def __aggregate_analyzers(dicts: List[Dict[Constant, List[Searchable]]]) -> Dict[Constant, List[Searchable]]:
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
