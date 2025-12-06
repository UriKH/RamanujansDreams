from dataclasses import dataclass, field
import sympy as sp
import mpmath as mp
from typing import Union, Dict, Optional, Any

from dreamer.utils.caching import cached_property


class Constant:
    registry: Dict[str, 'Constant'] = dict()

    def __init__(
            self,
            name: str,
            value_sympy: sp.Expr,
            value_mpmath: Optional[Union[mp.mpf, mp.mpc]] = None
    ):
        self.name = name
        self.value_sympy = value_sympy

        if value_mpmath:
            self.value_mpmath = value_mpmath
        Constant.registry[self.name] = self

    @cached_property
    def value_mpmath(self):
        if self.value_mpmath:
            return self.value_mpmath
        else:
            try:
                return sp.lambdify([], self.value_sympy, modules=['mpmath'])()
            except Exception as e:
                print(f"Warning: Could not auto-convert {self.name} to mpmath. Error: {e}")
                return mp.mpf(0)

    def __mul__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{self.name}*{other.name}', self.value_sympy * other.value_sympy)
        if isinstance(other, int):
            return Constant(f'{self.name}*{other}', self.value_sympy * other)
        raise TypeError(f"Unsupported operand for __mul__: '{type(other)}'")

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{self.name}+{other.name}', self.value_sympy + other.value_sympy)
        if isinstance(other, int):
            return Constant(f'{self.name}+{other}', self.value_sympy + other)
        raise TypeError(f"Unsupported operand for __add__: '{type(other)}'")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{self.name}-{other.name}', self.value_sympy - other.value_sympy)
        if isinstance(other, int):
            return Constant(f'{self.name}-{other}', self.value_sympy - other)
        raise TypeError(f"Unsupported operand for __sub__: '{type(other)}'")

    def __rsub__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{other.name}-{self.name}', other.value_sympy - self.value_sympy)
        if isinstance(other, int):
            return Constant(f'{other}-{self.name}', other - self.value_sympy)
        raise TypeError(f"Unsupported operand for __rsub__: '{type(other)}'")

    @staticmethod
    def is_registered(name: str) -> bool:
        return name in Constant.registry

    @staticmethod
    def available_constants():
        return list(Constant.registry.keys())

    @staticmethod
    def get_constant(name: str) -> 'Constant':
        return Constant.registry[name]

    def __hash__(self):
        return hash(self.name)
