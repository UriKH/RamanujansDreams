from dreamer import System, config
from dreamer import analysis_stage, search_stage
from dreamer.db_stage import *
import sympy as sp

if __name__ == '__main__':
    config.configure(
        system={
            'EXPORT_CMFS': './mycmfs',                          # export CMF as objects to directory: ./mycmfs
            'EXPORT_ANALYSIS_PRIORITIES': './myshards',         # export shards found in analysis into: ./myshards
            'EXPORT_SEARCH_RESULTS': './mysearchresults'        # export the search results into: ./mysearchresults
        },
        analysis={
            'IDENTIFY_THRESHOLD': 0,            # ignore shards with less than 20% identified trajectories as converge
                                                # to the constant
            'PARTIAL_SEARCH_FACTOR': 0.1        # out of auto-generated trajectories, use only 30% in analysis
        }
    )
    System(
        if_srcs=[pFq_formatter('pi', 2, 1, sp.Rational(1, 2), [0, 0, sp.Rational(1, 2)])],
        analyzers=[analysis_stage.AnalyzerModV1],
        searcher=search_stage.SearcherModV1
    ).run(constants=['pi'])
