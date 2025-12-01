from rt_search import *
from rt_search.db_stage.funcs.pFq_fmt import pFq_formatter
import sympy as sp


if __name__ == '__main__':
    config.configure(
        system={
            'EXPORT_CMFS': './mycmfs',
            'EXPORT_ANALYSIS_PRIORITIES': './mypriorities',
            'EXPORT_SEARCH_RESULTS': './mysearchresults'
        },
        analysis={
            'IDENTIFY_THRESHOLD': 0,
            'PARTIAL_SEARCH_FACTOR': 0.2
        },
        search={'PARALLEL_SEARCH': True}
    )
    results = System(
        if_srcs=[
            pFq_formatter(
                'pi', 3, 1, sp.Rational(1, 2),
                [sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)]
            )
        ],
        analyzers=[AnalyzerModV1],
        searcher=SearcherModV1
    ).run(constants=['pi'])
