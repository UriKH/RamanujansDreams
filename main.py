import mpmath as mp
import dreamer.loading
from dreamer import System, config
from dreamer import analysis, search
from dreamer.extraction.extractor import ShardExtractorMod
from dreamer.loading import *
import sympy as sp
from dreamer import pi, zeta


mp.dps = 300

if __name__ == '__main__':
    trajectory_compute_func = (lambda d: max(100, 10))

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
            'PARALLEL_SEARCH': True,
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func
            # number of trajectories to be auto generated in search if needed by the module
        }
    )

    # dreamer.loading.DBModScheme.export_future_append_to_json(
    #     [
    #         pFq_formatter(pi, 2, 1, sp.Rational(1, 2), [0, 0, sp.Rational(1, 2)]),
    #         pFq_formatter(pi, 3, 2, sp.Rational(1, 2), [sp.Rational(1, 2)] * 5)
    #     ], exits_ok=True
    # )
    from dreamer import Constant

    from sympy import symbols, summation, oo

    # n = symbols('n', integer=True, positive=True)
    #
    # def chi_minus_3(k):
    #     r = k % 3
    #     if r == 0:
    #         return 0
    #     elif r == 1:
    #         return 1
    #     else:
    #         return -1
    #
    # L2_chi_minus_3 = summation(chi_minus_3(n) / n ** 2, (n, 1, oo))
    #
    # calagari = Constant('clageri', L2_chi_minus_3)
    System(
        if_srcs=[ #pFq_formatter(pi, 2, 1, -1, [0, 0, 0]),
                 pFq_formatter(pi, 2, 1, -1, [0, 0, 0])
                 # pFq_formatter(pi, 2, 1, -1, [sp.Rational(1, 2)] * 3)
                 ],
        extractor=ShardExtractorMod,
        # extractor=None,
        analyzers=[analysis.AnalyzerModV1],
        searcher=search.SearcherModV1
    ).run(constants=[pi])
