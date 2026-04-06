from collections import defaultdict
import networkx as nx
from itertools import combinations
import os

from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.schemes.db_scheme import DBModScheme
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.extraction_scheme import ExtractionModScheme
from dreamer.utils.storage import Exporter, Importer, Formats
from dreamer.utils.types import *
from dreamer.utils.logger import Logger
from dreamer.utils.constants.constant import Constant
from dreamer.configs import config
from functools import partial
import sympy as sp

sys_config = config.system
extraction_config = config.extraction


class System:
    """
    System class wraps together all given modules and connects them.
    """

    def __init__(self,
                 if_srcs: List[DBModScheme | str | Formatter],
                 extractor: Optional[Type[ExtractionModScheme]],
                 analyzers: List[Type[AnalyzerModScheme] | partial[AnalyzerModScheme] | str | Searchable],
                 searcher: Type[SearcherModScheme] | partial[SearcherModScheme]):
        """
        Constructing a system runnable instance for a given combination of modules.
        :param if_srcs: A list of DBModScheme instances used as sources.
        :param extractor: An optional ExtractionModScheme type used to extract shards from the CMFs.
        If extractor not provided, analysis will try to read from the default searchables directory.
        :param analyzers: A list of AnalyzerModScheme types used for prioritization + preparation before the search
        :param searcher: A SearcherModScheme type used to deepen the search done by the analyzers
        """
        if not isinstance(if_srcs, list):
            raise ValueError(f'Inspiration Functions must be contained in a list')

        self.if_srcs = if_srcs
        self.extractor = extractor
        self.analyzers = analyzers
        self.searcher = searcher

        if not self.if_srcs and self.extractor:
            raise ValueError(f'Could not preform extraction if no sourced to extract from where provided')

    def run(self, constants: Optional[List[str | Constant] | str | Constant] = None):
        """
        Run the system given the constants to search for.
        :param constants: if None, search for constants defined in the configuration file in 'configs.database.py'.
        """
        char = '='
        total = 150
        Logger(char * total, Logger.Levels.debug).log()
        Logger(Logger.buffer_print(total, 'NEW RUN', char), Logger.Levels.debug).log()
        Logger(char * total, Logger.Levels.debug).log()

        constants = self.__validate_constants(constants)

        # ======================================================
        # LOAD STAGE - loading constants (and optional storage)
        # ======================================================
        cmf_data = self.__loading_stage(constants)

        if path := sys_config.EXPORT_CMFS:
            os.makedirs(path, exist_ok=True)

            for const, l in cmf_data.items():
                safe_key = "".join(c for c in const.name if c.isalnum() or c in ('-', '_'))
                const_path = os.path.join(path, safe_key)

                for scmf in l:
                    if scmf.cmf.__class__ == CMF:
                        filename = f'generated_cmf_hashed_{hash(scmf.cmf)}'
                    else:
                        filename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in repr(scmf.cmf)).strip('_')
                    Exporter.export(
                        root=const_path, f_name=filename, exists_ok=True, clean_exists=True,
                        data=[ShiftCMF(scmf.cmf, scmf.shift, True)], fmt=Formats.PICKLE
                    )
                Logger(
                    f'CMFs for {const.name} exported to {const_path}', Logger.Levels.info
                ).log()

        # print constants and CMFs
        for constant, funcs in cmf_data.items():
            functions = '\n'
            for i, func in enumerate(funcs):
                if func.cmf.__class__ == CMF:
                    formatted_matrices = []
                    for sym, mat in func.cmf.matrices.items():
                        # Keep the symbol (like 'A' or 'B') pretty
                        sym_str = sp.pretty(sym, use_unicode=True)

                        # Format the matrix as a standard string, row by row,
                        # completely avoiding the broken 2D ASCII brackets
                        rows_str = ',\n'.join(f'    {str(list(mat.row(r)))}' for r in range(mat.rows))
                        mat_str = f"[\n{rows_str}\n  ]"

                        formatted_matrices.append(f"{sym_str}:\n  {mat_str}")

                    pretty_mats = '\n\n>>> '.join(formatted_matrices)
                    # pretty_mats = '\n\n>>> '.join(f'{sp.pretty(sym, use_unicode=True, wrap_line=False)}:\n{sp.pretty(mat, use_unicode=True, wrap_line=False)}' for sym, mat in func.cmf.matrices.items())
                    functions += f'{i+1}. CMF: \n>>>{pretty_mats}\n with offset {tuple(func.shift.values())}\n'
                else:
                    functions += f'{i+1}. CMF: {repr(func.cmf)} with offset {tuple(func.shift.values())}\n'
            Logger(
                f'Searching for {constant.name} using inspiration functions: {functions}', Logger.Levels.info
            ).log()

        # ====================================================
        # EXTRACTION STAGE - computing shards and saving them
        # ====================================================
        shard_dict = dict()
        if self.extractor:
            shard_dict = self.extractor(cmf_data).execute()
        else:
            # Load saved shards from the default directory if data not provided
            for const_name in os.listdir(extraction_config.PATH_TO_SEARCHABLES):
                import_stream = Importer.import_stream(f'{extraction_config.PATH_TO_SEARCHABLES}\\{const_name}')
                const_shards = []
                for shards in import_stream:
                    const_shards += shards
                if const_shards:
                    shard_dict[const_shards[0].const] = const_shards

        # =======================================================
        # ANALYSIS STAGE - analyzes shards and prioritize search
        # =======================================================
        priorities = self.__analysis_stage(shard_dict)
        Logger.timer_summary()

        # Store priorities to be used in the search stage and future runs
        filtered_priorities = dict()
        bad_run = False
        if path := sys_config.EXPORT_ANALYSIS_PRIORITIES:
            os.makedirs(path, exist_ok=True)

            for const, l in priorities.items():
                if not l:
                    Logger(
                        f'No shards remained after analysis. Run for constant "{const.name}" is stopped.',
                        Logger.Levels.warning
                    ).log()
                    continue

                safe_key = "".join(c for c in const.name if c.isalnum() or c in ('-', '_'))
                const_path = os.path.join(path, safe_key)

                Exporter.export(const_path, exists_ok=True, clean_exists=False, data=l, fmt=Formats.PICKLE)
                Logger(
                    f'Priorities for {const.name} exported to {const_path}', Logger.Levels.info
                ).log()
                filtered_priorities[const] = l

            if not filtered_priorities:
                bad_run = True

        if bad_run or not priorities:
            Logger('No relevant shards found, run stopped', Logger.Levels.warning).log()
            return

        # =======================================================
        # SEARCH STAGE - preform deep search
        # =======================================================
        if len(filtered_priorities) == 0:
            filtered_priorities = priorities
        self.__search_stage(filtered_priorities)

    def __loading_stage(self, constants: List[Constant]) -> Dict[Constant, List[ShiftCMF]]:
        """
        Preforms the loading of the inspiration functions from various sources
        :param constants: A list of all constants relevant to this run
        :return: A mapping from a constant to the list of its CMFs (matching the inspiration functions)
        """
        if not self.if_srcs:
            return dict()

        Logger('Loading CMFs ...', Logger.Levels.info).log()
        modules = []
        cmf_data = defaultdict(set)

        for db in self.if_srcs:
            if isinstance(db, DBModScheme):
                modules.append(db)
            elif isinstance(db, str):
                shift_cmf = Importer.imprt(db)
                cmf_data[Constant.get_constant(db.split('/')[-2])].add(shift_cmf[0])
            elif isinstance(db, Formatter):
                cmf_data[Constant.get_constant(db.const)].add(db.to_cmf())
            else:
                raise ValueError(f'Unknown format: {db} (accepts only str | DBModScheme | Formatter)')

        # If DB were used, aggregate extracted constants
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
                    f'constant {k} is not in the search list, its inspiration function(s) will be ignored',
                    level=Logger.Levels.warning
                ).log()
                continue
            as_list[k] = list(v)
        return as_list

    def __analysis_stage(
            self, cmf_data: Optional[Dict[Constant, List[Searchable]]] = None
    ) -> Dict[Constant, List[Searchable]]:
        """
        Preform the analysis stage work
        :param cmf_data: data produced in the loading stage
        :return: The results of the analysis as a mapping from constant to a list of prioritized searchables.
        """
        analyzers: List[Type[AnalyzerModScheme]] = []
        results = defaultdict(set)

        # prepare analyzers
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

        # Load saved shards from the default directory if data not provided
        # if not cmf_data:
        #     cmf_data = {}
        #     for const_name in os.listdir(extraction_config.PATH_TO_SEARCHABLES):
        #         import_stream = Importer.import_stream(f'{extraction_config.PATH_TO_SEARCHABLES}\\{const_name}')
        #         const_shards = []
        #         for shards in import_stream:
        #             const_shards += shards
        #         if const_shards:
        #             cmf_data[const_shards[0].const] = const_shards

        analyzers_results = [analyzer(cmf_data).execute() for analyzer in analyzers]
        priorities = self.__aggregate_analyzers(analyzers_results)

        # add unprioritized elements to the end
        for c, l in results.items():
            if c not in priorities:
                priorities[c] = list(l)
            else:
                diff = l.difference(set(cmf_data[c]))
                priorities[c].extend(diff)
        return priorities

    def __search_stage(self, priorities: Dict[Constant, List[Searchable]]):
        """
        Preform deep search using the provided search module
        :param priorities: a list prioritized searchables for each constant
        """
        # Execute searchers
        for data in priorities.values():
            self.searcher(data, sys_config.USE_LIReC).execute()

        # Print best delta for each constant
        for const in priorities.keys():
            best_delta = -sp.oo
            best_sv = None
            dir_path = os.path.join(sys_config.EXPORT_SEARCH_RESULTS, const.name)

            # TODO: we first need to read inside the directories
            stream_gen = Importer.import_stream(dir_path)
            for dm in stream_gen:
                delta, sv = dm.best_delta
                if delta is None:
                    continue
                if best_delta < delta:
                    best_delta, best_sv = delta, sv

            if best_sv is None:
                # Should not happen
                Logger('No best delta found').log()
            else:
                Logger(
                    f'Best delta for "{const.name}" found by the searcher is {best_delta}\n'
                    f'* Trajectory: {best_sv.trajectory} \n* Start: {best_sv.start}',
                    Logger.Levels.info
                ).log()

        # delete temp directory
        if sys_config.EXPORT_SEARCH_RESULTS.split('.')[-1] == sys_config.DEFAULT_DIR_SUFFIX:
            os.rmdir(sys_config.EXPORT_SEARCH_RESULTS)

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

    @staticmethod
    def __validate_constants(constants: Optional[List[str | Constant] | str | Constant] = None) -> List[Constant]:
        """
        Validates constants are in the correct format and usable
        :param constants: One or more Constant object or a constant name
        :return: A list of Constant objects
        """
        if not constants:
            Logger(
                'No constants provided, searching for all constants in configurations', Logger.Levels.warning
            ).log()
            constants = sys_config.CONSTANTS

        # prepare constants for loading
        if isinstance(constants, str | Constant):
            constants = [constants]
        as_obj = []
        for c in constants:
            if isinstance(c, str):
                if not Constant.is_registered(c):
                    raise ValueError(f'Constant "{c}" is not a registered constant.')
                as_obj.append(Constant.get_constant(c))
            else:
                as_obj.append(c)
        return as_obj

