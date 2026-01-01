from abc import ABC, abstractmethod

import numpy as np
from ramanujantools import Position, Limit
from ramanujantools.cmf import CMF
import ramanujantools as rt
from typing import Optional, Tuple, Set
from LIReC.db.access import db
from dreamer.utils.constants.constant import Constant
import mpmath as mp

from dreamer.utils.logger import Logger
from dreamer.utils.storage.storage_objects import SearchData, SearchVector
import sympy as sp
from dreamer.configs.search import search_config


n = sp.symbols('n')


class Searchable(ABC):
    def __init__(self, cmf: CMF, constant: Constant, shift: Position):
        self.cache = []
        self.cmf = cmf
        self.const = constant
        self.shift = shift

    @abstractmethod
    def in_space(self, point: Position) -> bool:
        raise NotImplementedError()

    @property
    def dim(self):
        return self.cmf.dim()

    def calc_delta(self, traj_m, constant: sp.Expr, traj_len: float) \
            -> Tuple[Optional[float], Optional[rt.Matrix], Optional[float]]:
        """
        Computes delta for a given trajectory, start point and constant.
        :param cmf: the CMF to compute in.
        :param traj: trajectory from the start point
        :param start: start point to compute walk from.
        :param constant: the constant to compute delta for.
        :return: the pair: delta, estimated_value.
        """
        # Do walk
        # with Logger.simple_timer('walk'):
        try:
            walked = traj_m.walk({n: 1}, search_config.DEPTH_FROM_TRAJECTORY_LEN(traj_len) , {n: 0})
            walked = walked.inv().T
        except Exception as e:
            Logger(f'Unexpected exception when trying to walk, ignoring trajectory', Logger.Levels.warning).log(msg_prefix='\n')
            return None, None, None
        t1_col = (walked / walked[0, 0]).col(0)

        # Cache lookup
        p, q = None, None
        values = [item for item in t1_col]
        pi_30000 = constant.evalf(30000)
        cache_hit = False
        values_vec = sp.Matrix(values)

        numerator, denom = None, None
        # with Logger.simple_timer('cache search'):
        for v1, v2 in self.cache:
            v1 = sp.Matrix(v1).T
            v2 = sp.Matrix(v2).T
            numerator = v1.dot(values_vec)
            denom = v2.dot(values_vec)
            estimated = sp.Abs(sp.Rational(numerator, denom))
            err = sp.Abs(estimated - pi_30000)
            if sp.N(err, 25) < 1e-12:
                p, q = v1, v2
                cache_hit = True
                break

        # If cache miss - use LIReC
        # with Logger.simple_timer('cache miss'):
        if not cache_hit:
            try:
                # with Logger.simple_timer('Identify'):
                res = db.identify([constant.evalf(300)] + t1_col[1:])
            except Exception as e:
                # print(f'traj={traj_m}, constant={constant}, {e}')
                # return None, None, None
                raise Exception(f'LIReC failed with: {e}')

            if not res:
                return None, None, None

            res = res[0]
            res.include_isolated = 0
            estematedExpr = sp.nsimplify(str(res).rsplit(' ', 1)[0], rational=True)
            numerator, denom = sp.fraction(estematedExpr)
            p_dict = numerator.as_coefficients_dict()
            q_dict = denom.as_coefficients_dict()
            syms = sp.symbols(f'c:{traj_m.shape[0]}')[1:]
            p, q = [p_dict[sym] for sym in [1] + list(syms)], [q_dict[sym] for sym in [1] + list(syms)]

        # with Logger.simple_timer('check convergence'):
        # Check convergence
        try:
            converge, (_, limit, _) = self._does_converge(traj_m, p, q)
            converge = True
        except Exception as e:
            print(f'convergence exception: {e}')
            converge = False

        if not converge:
            return None, None, None

        # Add to cache
        if not cache_hit:
            self.cache.append((tuple(p), tuple(q)))

        # with Logger.simple_timer('delta computation'):
        # estimate constant
        p = sp.Matrix(p).T
        q = sp.Matrix(q).T
        if not cache_hit:
            numerator = p.dot(values_vec)
            denom = q.dot(values_vec)
        estimated = sp.Abs(sp.Rational(numerator, denom))

        # check abnormal denominator and compute delta
        err = sp.Abs(estimated - pi_30000)
        denom = sp.denom(estimated)
        if denom == 1:
            # raise ZeroDivisionError('Denominator 1 caused zero division in delta calculation')
            # Logger(f'Denominator 1 caused zero division in delta calculation',
            #        Logger.Levels.warning).log()
            return None, None, None
        if denom < 1e6:
            # raise Exception(f"Probably still rational as denominator is quite small: {denom}")
            # Logger(f"Probably still rational as denominator is quite small: {denom}",
            #        Logger.Levels.warning).log()
            return None, None, None

        delta = -1 - sp.log(err) / sp.log(denom)
        if delta == sp.oo or delta == sp.zoo:
            if err == 0:
                Logger(f'delta guardrails failed, got delta={delta} with: error=0',
                       Logger.Levels.warning).log(msg_prefix='\n')
            if denom == 0:
                Logger(f'delta guardrails failed, got delta={delta} with: denom=0',
                       Logger.Levels.warning).log(msg_prefix='\n')
            if denom != 0 and err != 0:
                Logger(f'delta guardrails failed, got delta={delta} with: \nerror={err} \ndenom = {denom}', Logger.Levels.warning).log(msg_prefix='\n')
            return None, None, None

        return float(delta.evalf(10)), rt.Matrix([p, q]), float(limit.as_float())

    def compute_trajectory_data(self, traj: Position, start: Position,
                                *, find_limit: bool = False,
                                find_eigen_values: bool = False,
                                find_gcd_slope: bool = False,
                                use_LIReC: bool = True) -> SearchData:
        """
        Compute delta search results for a given trajectory, start point and constant.
        :param traj: The trajectory to search for.
        :param start: Start point of search.
        :param constant: Name of the constant in sympy format (using LIReC) / mpmath format (using RT identify).
        :param find_limit: Compute the limit of the trajectory matrix.
        :param find_eigen_values: Compute the eigenvalues of the trajectory matrix.
        :param find_gcd_slope: Compute the GCD slope.
        :param use_LIReC: Use LIReC to compute delta (default)
        :return: SearchData object containing the results of the search.
        """
        # with Logger.simple_timer('compute_trajectory_matrix'):
        sd = SearchData(SearchVector(start, traj))
        try:
            traj_m = self.cmf.trajectory_matrix(
                trajectory=traj,
                start=start
            )
        except Exception as e:
            Logger(f'error while computing trajectory matrix for start={start}, trajectory={traj}: {e}', Logger.Levels.warning).log(msg_prefix='\n')
            return sd

        if find_limit:
            limit = traj_m.limit({n: 1}, 2000, {n: 0})
            sd.limit = float(limit.as_float())
        if find_eigen_values:
            sd.ev = traj_m.eigenvals()
        if find_gcd_slope:
            sd.gcd_slope = traj_m.gcd_slope()
            sd.gcd_slope = float(sd.gcd_slope)

        if not use_LIReC and find_limit:
            # with Logger.simple_timer('compute_limit - no LIReC'):
            sd.initial_values = limit.identify(self.const.value_mpmath)
            sd.delta = limit.delta(self.const.value_mpmath)
            if sd.delta in (mp.mpf("inf"), mp.mpf("-inf")):  # TODO: delta is a float!
                sd.delta = str(sd.delta)
        else:
            # with Logger.simple_timer('compute_limit - LIReC'):
            if not use_LIReC and not find_limit:
                print('in order to compute delta must find limit - defaulting to using LIReC')
            sd.delta, sd.initial_values, sd.limit = self.calc_delta(
                traj_m, self.const.value_sympy, np.sqrt(np.sum(np.array(list(traj.values()), dtype=np.float64) ** 2)).astype(float)
            )
            if sd.delta is not None:
                sd.LIReC_identify = True
        return sd

    @staticmethod
    def _does_converge(t_mat: rt.Matrix, p, q) -> Tuple[bool, Tuple[Limit, Limit, Limit]]:
        l1, l2, l3 = t_mat.limit(
            {n: 1}, [950, 1000, 1050], {n: 0}
        )

        floats = []
        l = [l1, l2, l3]
        for lim in l:
            mat = lim.current.inv().T
            t1_col = (mat / mat[0, 0]).col(0)
            values = [item for item in t1_col]
            values_vec = sp.Matrix(values)
            p = sp.Matrix(p).T
            q = sp.Matrix(q).T
            numerator = p.dot(values_vec)
            denom = q.dot(values_vec)
            estimated = sp.Abs(sp.Rational(numerator, denom))
            floats.append(estimated)

        f1, f2, f3 = floats
        diff1 = abs(f1 - f2)
        diff2 = abs(f3 - f2)
        if diff1 < 1e-10 and diff2 < 1e-10:
            return True, (l1, l2, l3)
        return False, (l1, l2, l3)

    @abstractmethod
    def get_interior_point(self) -> Position:
        raise NotImplementedError()

    @abstractmethod
    def sample_trajectories(self, n_samples: int, *, strict: Optional[bool] = False) -> Set[Position]:
        raise NotImplementedError()

