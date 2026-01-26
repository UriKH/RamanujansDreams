from abc import ABC, abstractmethod
import numpy as np
from LIReC.db.access import db
import mpmath as mp
import ramanujantools as rt
from ramanujantools import Limit

from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.storage.storage_objects import SearchData, SearchVector
from dreamer.configs.search import search_config
from dreamer.utils.types import *
from dreamer.utils.storage.frequency_list import FrequencyList


n = sp.symbols('n')


class Searchable(ABC):
    """
    A template for a general searchable object (e.g., shards)
    """

    def __init__(self, cmf: CMF, constant: Constant, shift: Position):
        """
        :param cmf: The CMF to search in.
        :param constant: A constant to search for.
        :param shift: The shift in the starting point of the CMF.
        """
        self.cache = FrequencyList(max_size=100)
        self.cmf = cmf
        self.const = constant
        self.shift = shift

    def __repr__(self):
        return f'{self.__class__.__name__}(cmf={self.cmf}, constant={self.const}, shift={self.shift})'

    def __str__(self):
        return f'{self.cmf}_shift={tuple(self.shift.values())}_start={tuple(self.get_interior_point().values())}'

    @abstractmethod
    def in_space(self, point: Position) -> bool:
        """
        Checks if a point is inside the searchable object.
        :param point: A point to check if it is inside the searchable object
        :return: True if the point is inside the searchable object, false otherwise.
        """
        raise NotImplementedError()

    @property
    def dim(self):
        """
        :return: Space (CMF) dimension.
        """
        return self.cmf.dim()

    def calc_delta(self, traj_m, constant: sp.Expr, traj_len: float) \
            -> Tuple[Optional[float], Optional[rt.Matrix], Optional[float]]:
        """
        Computes delta for a given trajectory, start point, and constant.
        :param traj_m: The trajectory matrix.
        :param constant: The constant to compute delta for.
        :param traj_len: The length of the trajectory vector.
        :return: The pair (delta, estimated_value).
        """
        # Do walk
        try:
            with Logger.simple_timer('Initial walk and inverse'):
                walked = traj_m.walk({n: 1}, search_config.DEPTH_FROM_TRAJECTORY_LEN(traj_len), {n: 0})
                walked = walked.inv().T
        except Exception as e:
            Logger(f'Unexpected exception when trying to walk, ignoring trajectory {e}', Logger.Levels.warning).log(msg_prefix='\n')
            return None, None, None
        t1_col = (walked / walked[0, 0]).col(0)

        # initial values
        p, q = None, None
        values = [item for item in t1_col]
        with Logger.simple_timer('constant heavy evalf'):
            pi_30000 = constant.evalf(30000)
            pi_300 = constant.evalf(300)
        cache_hit = False
        values_vec = sp.Matrix(values)
        estimated = None
        err = None

        # Cache lookup
        with Logger.simple_timer('cache lookup'):
            def matcher(v):
                v1, v2 = v
                v1 = sp.Matrix(v1).T
                v2 = sp.Matrix(v2).T
                numerator = v1.dot(values_vec)
                denom = v2.dot(values_vec)
                err = sp.Abs(sp.Abs(sp.Rational(numerator, denom)) - pi_300)
                return sp.N(err, 25) < search_config.CACHE_ACCEPTANCE_THRESHOLD

            matched = self.cache.find(matcher)
            if matched:
                p, q = matched
                cache_hit = True

        # If cache misses - use LIReC
        if not cache_hit:
            try:
                with Logger.simple_timer('LIReC identify'):
                    res = db.identify([pi_300] + t1_col[1:])
            except Exception as e:
                # LIReC might fail for some reason like tolerance or something else.
                # This is not expected to occur but could happen nonetheless and should be reported to the user.
                # User should probably change the "depth from trajectory"
                var_name = f'{search_config.DEPTH_FROM_TRAJECTORY_LEN=}'.split('=')[0]
                Logger(
                    f'Note that LIReC failed with "{e}"\n'
                    f'This is probably an issue with the current '
                    f'{var_name} configuration',
                    Logger.Levels.warning
                ).log(msg_prefix='\n')
                return None, None, None

            # if LIReC failed to identify the constant
            if not res:
                return None, None, None

            # extract p,q vectors
            with (Logger.simple_timer('LIReC identify - postprocessing')):
                res = res[0]
                res.include_isolated = 0
                estimated_expr = sp.nsimplify(str(res).rsplit(' ', 1)[0], rational=True)
                numerator, denom = sp.fraction(estimated_expr)
                p_dict = numerator.as_coefficients_dict()
                q_dict = denom.as_coefficients_dict()
                syms = sp.symbols(f'c:{traj_m.shape[0]}')[1:]
                ext_syms = [1] + list(syms)
                p, q = [p_dict[sym] for sym in ext_syms], [q_dict[sym] for sym in ext_syms]

                # check convergence to constant
                estimated = estimated_expr.subs({sym: v for sym, v in zip(ext_syms, list(values_vec))})
                err = sp.Abs(estimated - pi_30000)
                if sp.N(err, 15) > search_config.IDENTIFY_CHECK_THRESHOLD:
                    return None, None, None

        # Check path convergence
        try:
            with Logger.simple_timer('convergence check'):
                converge, (_, limit, _) = self._does_converge(traj_m, traj_len, p, q)
        except Exception as e:
            print(f'convergence exception: {e}')
            converge = False
        if not converge:
            return None, None, None

        # Add p,q vectors to the cache
        if not cache_hit:
            self.cache.append((tuple(p), tuple(q)))

        # Estimate constant
        with Logger.simple_timer('estimate constant and delta compute'):
            p = sp.Matrix(p).T
            q = sp.Matrix(q).T
            if not estimated or not err:
                numerator = p.dot(values_vec)
                denom = q.dot(values_vec)
                estimated = sp.Abs(sp.Rational(numerator, denom))
                err = sp.Abs(estimated - pi_30000)

            # check abnormal denominator and compute delta
            denom = sp.denom(estimated)
            if sp.Abs(denom) <= search_config.MIN_ESTIMATE_DENOMINATOR:
                # probably didn't converge for some reason
                return None, None, None

            delta = -1 - sp.log(err) / sp.log(denom)

            # This part is not supposed to be reached at all, these are the final guardrails
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

        traj_len = np.sqrt(np.sum(np.array(list(traj.values()), dtype=np.float64) ** 2)).astype(float)
        if find_limit:
            limit = traj_m.limit({n: 1}, search_config.DEPTH_FROM_TRAJECTORY_LEN(traj_len), {n: 0})
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
                traj_m, self.const.value_sympy, traj_len
            )
            if sd.delta is not None:
                sd.LIReC_identify = True
        return sd

    @staticmethod
    def _does_converge(t_mat: rt.Matrix, traj_len, p, q) -> Tuple[bool, Tuple[Limit, Limit, Limit]]:
        """
        Checks if the trajectory matrix converges to the constant using the p, q vectors.
        :param t_mat: The trajectory matrix to compute limits for.
        :param traj_len: The length of the trajectory vector.
        :param p: p vector
        :param q: q vector
        :return: True if the trajectory matrix converges, false otherwise.
        """
        # compute limits
        depth = search_config.DEPTH_FROM_TRAJECTORY_LEN(traj_len)
        l1, l2, l3 = t_mat.limit(
            {n: 1},
            [round(coef * depth) for coef in search_config.DEPTH_CONVERGENCE_THRESHOLD],
            {n: 0}
        )
        floats = []
        l = [l1, l2, l3]

        # transform to floats
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

        # check diffs
        if diff1 < search_config.LIMIT_DIFF_ERROR_BOUND and diff2 < search_config.LIMIT_DIFF_ERROR_BOUND:
            return True, (l1, l2, l3)
        return False, (l1, l2, l3)

    @abstractmethod
    def get_interior_point(self) -> Position:
        """
        :return: A point inside the searchable
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_trajectories(self, n_samples: int, *, strict: Optional[bool] = False) -> Set[Position]:
        """
        Sample trajectories from the searchable.
        :param n_samples: Number of trajectories in searchable or number of samples to generate (depends on 'strict').
        :param strict: If true, sample exactly n_samples trajectories from the searchable,
            else sample n_samples * fraction.
        :return: A set of sampled trajectories
        """
        raise NotImplementedError()

