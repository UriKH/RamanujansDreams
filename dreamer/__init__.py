from . import loading, analysis, search, system
from .system import System

# # configs
from .configs import (
    config,
    sys_config,
    db_config,
    analysis_config,
    search_config
)


from dreamer.utils.constants.constant import Constant
import sympy as sp


e = Constant('e', sp.E)
pi = Constant('pi', sp.pi)
euler_gamma = Constant('euler_gamma', sp.EulerGamma)
pi_squared = Constant('pi_squared', sp.pi ** 2)
catalan = Constant('catalan', sp.Catalan)
gompertz = Constant('gompertz', -sp.exp(1) * sp.Ei(-1))


def zeta(n):
    if f'zeta_{n}' not in Constant.registry:
        return Constant(f'zeta_{n}', sp.zeta(n))
    return Constant.registry[f'zeta_{n}']


def sqrt(v):
    if isinstance(v, Constant):
        return Constant(f'sqrt({v.name})', sp.sqrt(v.value_sympy))
    if isinstance(v, int):
        return Constant(f'sqrt({v})', sp.sqrt(v))
    raise TypeError(f"Unsupported operand for sqrt: '{type(v)}'")


def power(v: Constant, n: int):
    return Constant(f'{v.name}^{n}', v.value_sympy ** n)


def log(n):
    if f'log_{n}' not in Constant.registry:
        return Constant(f'log_{n}', sp.log(n))
    return Constant.registry[f'log_{n}']

