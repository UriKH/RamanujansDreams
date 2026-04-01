from dreamer import System, config
from dreamer import analysis, search, extraction
from dreamer.loading import *
from dreamer import zeta, log, pi
import sympy as sp


# Because of pickling format we need to define these functions here
def trajectory_compute_func(d):
    return max(10 ** d, 10)


def trajectory_compute_func_analysis(d):
    return max(10 ** d * 2, 10)


if __name__ == '__main__':

    config.configure(
        system={
            'EXPORT_CMFS': './mycmfs',                          # export CMF as objects to directory: ./mycmfs
            'EXPORT_ANALYSIS_PRIORITIES': './myshards',         # export shards found in analysis into: ./myshards
            'EXPORT_SEARCH_RESULTS': './mysearchresults'        # export the search results into: ./mysearchresults
        },
        analysis={
            # ignore shards with less than 0.1% identified trajectories as converge to the constant
            'IDENTIFY_THRESHOLD': 1e-3,
            # number of trajectories to be auto-generated in analysis
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func_analysis
        },
        extraction={
            'INIT_POINT_MAX_COORD': 10,
            # In this case this indicates usage of pFq symmetries utilization to reduce the number of shards
            'IGNORE_DUPLICATE_SEARCHABLES': True
        },
        search={
            # number of trajectories to be auto-generated in search if needed by the module
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func,
            'DEFAULT_USES_INV_T': False
        }
    )

    p = 2
    q = 1
    z = -1


    System(
        if_srcs=[pFq(log(2), p, q, z)],
        extractor=extraction.extractor.ShardExtractorMod,
        analyzers=[analysis.AnalyzerModV1],
        searcher=search.SearcherModV1
    ).run(constants=[log(2)])
