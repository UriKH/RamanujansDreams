import unittest
from contextlib import contextmanager

from itertools import product
from functools import lru_cache
from scipy.optimize import linprog
import numpy as np

from dreamer.utils import Plane
from dreamer.utils import *
from dreamer.configs.analysis import *

x0, x1, y0 = sp.symbols('x0 x1 y0')


@contextmanager
def safe(testcase, exc_type=Exception):
    try:
        yield
    except exc_type as e:
        testcase.fail(f"Unexpectedly raised {type(e).__name__}: {e}")


class TestDB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_encode_points(self):
        symbols = [x0, x1, y0]
        hps = [
            # Plane(x1 - y0 + 1, symbols),
            # Plane(x0 - y0 + 1, symbols),
            Plane(y0, symbols),
            Plane(x0, symbols),
            Plane(x1, symbols),
            Plane(-x0 + y0, symbols),
            Plane(-x1 + y0, symbols)
        ]

        encoded = set(get_encoded_shards(hps, symbols))
        self.assertEqual(18, len(encoded))
        shards = {
            # XY > 0, Z > 0
            (1, 1, 1, -1, -1),
            (1, 1, 1, 1, 1),
            (1, 1, 1, 1, -1),
            (1, 1, 1, -1, 1),
            # XY > 0, Z < 0
            (-1, 1, 1, -1, -1),
            # XZ > 0, Y < 0
            (1, 1, -1, -1, 1),
            (1, 1, -1, 1, 1),
            # ZY < 0, X > 0
            (-1, 1, -1, -1, 1),
            (-1, 1, -1, -1, -1),
            # XY < 0, Z> 0
            (1, -1, -1, 1, 1),
            # XYZ < 0
            (-1, -1, -1, -1, -1),
            (-1, -1, -1, 1, 1),
            (-1, -1, -1, 1, -1),
            (-1, -1, -1, -1, 1),
            # YZ > 0, X < 0
            (1, -1, 1, 1, 1),
            (1, -1, 1, 1, -1),
            # ZX < 0, Y > 0
            (-1, -1, 1, 1, -1),
            (-1, -1, 1, -1, -1)
        }
        self.assertEqual(shards, encoded)
        # self.assertEqual(encoded, [(1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, -1), (1, 1, 1, 1, 1, -1, 1), (1, 1, 1, 1, 1, -1, -1), (1, 1, 1, 1, -1, 1, 1), (1, 1, 1, 1, -1, 1, -1), (1, 1, 1, 1, -1, -1, 1), (1, 1, 1, 1, -1, -1, -1), (1, 1, 1, -1, 1, 1, 1), (1, 1, 1, -1, 1, 1, -1), (1, 1, 1, -1, 1, -1, 1), (1, 1, 1, -1, 1, -1, -1), (1, 1, 1, -1, -1, 1, 1), (1, 1, 1, -1, -1, 1, -1), (1, 1, 1, -1, -1, -1, 1), (1, 1, 1, -1, -1, -1, -1), (1, 1, -1, 1, 1, 1, 1), (1, 1, -1, 1, 1, 1, -1), (1, 1, -1, 1, 1, -1, 1), (1, 1, -1, 1, 1, -1, -1), (1, 1, -1, 1, -1, 1, 1), (1, 1, -1, 1, -1, 1, -1), (1, 1, -1, 1, -1, -1, 1), (1, 1, -1, 1, -1, -1, -1), (1, 1, -1, -1, 1, 1, 1), (1, 1, -1, -1, 1, 1, -1), (1, 1, -1, -1, 1, -1, 1), (1, 1, -1, -1, 1, -1, -1), (1, 1, -1, -1, -1, 1, 1), (1, 1, -1, -1, -1, 1, -1), (1, 1, -1, -1, -1, -1, 1), (1, 1, -1, -1, -1, -1, -1), (1, -1, 1, -1, -1, 1, -1), (1, -1, 1, -1, -1, -1, -1), (1, -1, -1, -1, 1, 1, -1), (1, -1, -1, -1, 1, -1, -1), (1, -1, -1, -1, -1, 1, -1), (1, -1, -1, -1, -1, -1, -1), (-1, 1, -1, 1, 1, -1, -1), (-1, 1, -1, 1, -1, -1, -1), (-1, 1, -1, -1, 1, 1, -1), (-1, 1, -1, -1, 1, -1, -1), (-1, 1, -1, -1, -1, 1, -1), (-1, 1, -1, -1, -1, -1, -1), (-1, -1, -1, -1, 1, 1, -1), (-1, -1, -1, -1, 1, -1, -1), (-1, -1, -1, -1, -1, 1, -1), (-1, -1, -1, -1, -1, -1, -1)])


def get_encoded_shards(hps, symbols) -> List[ShardVec]:
    """
    Compute the Shards as Shard vector identifiers
    :return: A list of the vector identifiers
    """
    @lru_cache(maxsize=128 if AnalysisConfig.USE_CACHING else 0)
    def expr_to_ineq(expr, greater_than_0: bool = True):
        """
        Prepare a linear expression of the form: ax + b > 0 \n
        to the form scipy.linprog() receives: -ax <= b - err \n
        (similarly for ax + b < 0: ax <= -b - err)
        :param expr: The expression to transform into the relevant inequality
        :param greater_than_0: indicates the format
        :return: the row matching the expression and the constant in the scipy.linprog() format
        """
        coeffs = expr.as_coefficients_dict()
        a = [float(coeffs.get(v, 0)) for v in symbols]
        const = float(expr.as_independent(*symbols, as_Add=True)[0])

        if greater_than_0:
            # a·x + c >= 0  ⇔  -a·x <= c
            row = [-coef for coef in a]
            b = const - SHARD_EXTRACTOR_ERR
        else:
            # a·x + c <= 0  ⇔  a·x <= -c
            row = [coef for coef in a]
            b = -const - SHARD_EXTRACTOR_ERR

        return row, b

    def validate_shard(shard: ShardVec) -> Tuple[bool, List[int | float]]:
        """
        Checks if a shard vector is valid in the CMF
        :param shard: the shard vector +-1's vector that describes the shard
        :return: True if a corresponding shard exists, else False
        """
        A, b = [], []
        for ineq, indicator in zip(hps, shard):
            row, rhs = expr_to_ineq(ineq.expression, indicator == 1)
            A.append(row)
            b.append(rhs)

        bounds = [(None, None)] * len(symbols)
        res = linprog(c=list(np.zeros(len(symbols))), bounds=bounds, A_ub=A, b_ub=b, method="highs")

        if not (res.success or res.status == 3):
            return False, []

        x = np.array(res.x, dtype=float) if res.x is not None else None
        eps = max(SHARD_EXTRACTOR_ERR, 1e-12)
        if x is None:
            return True, []

        # For each plane compute original value val = a·x + const
        # for ineq, indicator in zip(hps, shard):
        #     coeffs = ineq.expression.as_coefficients_dict()
        #     a = np.array([float(coeffs.get(v, 0)) for v in symbols], dtype=float)
        #     const = float(ineq.expression.as_independent(*symbols, as_Add=True)[0])
        #     val = a.dot(x) + const
        #     if indicator == 1:
        #         # require strict > 0
        #         if not (val > eps):
        #             return False, []
        #     else:
        #         # require strict < 0
        #         if not (val < -eps):
        #             return False, []

        # all strict inequalities passed
        return True, x.tolist()
        # return res.success, res.x.tolist() if res.success else []

    _encoded_shards = []
    for perm in product([+1, -1], repeat=len(hps)):
        valid, point = validate_shard(perm)
        if valid:
            _encoded_shards.append(perm)
    return _encoded_shards


if __name__ == "__main__":
    unittest.main()
