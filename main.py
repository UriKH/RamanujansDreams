import mpmath as mp
import dreamer.loading
from dreamer import System, config
from dreamer import analysis, search
from dreamer.extraction.extractor import ShardExtractorMod
from dreamer.loading import *
import sympy as sp
from dreamer import pi, zeta, log


mp.dps = 300

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
            'NUM_TRAJECTORIES_FROM_DIM': lambda d: 10 ** (d - 2)
            # number of trajectories to be auto generated in analysis
        },
        search={
            'PARALLEL_SEARCH': True,
            'NUM_TRAJECTORIES_FROM_DIM': lambda d: 10 ** (d - 1)
            # number of trajectories to be auto generated in search if needed by the module
        }
    )

    System(
        if_srcs=[
            pFq_formatter(
                dreamer.catalan, 3, 2, 1, [
                    0, 0, sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)
                ]
            ),
            pFq_formatter(
                dreamer.catalan, 4, 3, 1, [
                    0, 0, sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2), 0, 0
                ]
            )
        ],
        extractor=ShardExtractorMod,
        analyzers=[analysis.AnalyzerModV1],
        searcher=search.SearcherModV1
    ).run(constants=[dreamer.catalan])
