import mpmath as mp
import dreamer.loading
from dreamer import System, config
from dreamer import analysis, search
from dreamer.loading import *
import sympy as sp
from dreamer import pi, zeta



mp.dps = 300

if __name__ == '__main__':
    trajectory_compute_func = (lambda d: max(10 ** (d - 1), 10))

    config.configure(
        system={
            'EXPORT_CMFS': './mycmfs',                          # export CMF as objects to directory: ./mycmfs
            'EXPORT_ANALYSIS_PRIORITIES': './myshards',         # export shards found in analysis into: ./myshards
            'EXPORT_SEARCH_RESULTS': './mysearchresults'        # export the search results into: ./mysearchresults
        },
        analysis={
            'IDENTIFY_THRESHOLD': 0,            # ignore shards with less than 20% identified trajectories as converge
                                                # to the constant
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func
            # number of trajectories to be auto generated in analysis
        },
        search={
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func
            # number of trajectories to be auto generated in search if needed by the module
        }
    )

    # dreamer.db_stage.DBModScheme.export_future_append_to_json(
    #     [
    #         pFq_formatter(pi, 2, 1, sp.Rational(1, 2), [0, 0, sp.Rational(1, 2)]),
    #         pFq_formatter(pi, 3, 2, sp.Rational(1, 2), [sp.Rational(1, 2)] * 5)
    #     ], exits_ok=True
    # )

    System(
        if_srcs=[pFq_formatter(pi, 2, 1, sp.Rational(1, 2), [0, 0, sp.Rational(1, 2)])],
        analyzers=[analysis.AnalyzerModV1],
        searcher=search.SearcherModV1
    ).run(constants=[pi])
