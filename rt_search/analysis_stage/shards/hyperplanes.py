from ramanujantools import Position
from dataclasses import dataclass
from rt_search.utils.caching import *

import sympy as sp
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class Hyperplane:
    """
    Represents a hyperplane as a sympy expression.
    The expression might miss some symbols as the space the hyperplane lives in consists more axis than defined.
    """
    expr: sp.Expr
    symbols: Optional[List[sp.Basic]] = None

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = list(self.expr.free_symbols)
        if not self.expr.free_symbols.issubset(self.symbols):
            raise ValueError(
                f'Missing symbols in ordering. Expression contains {self.expr.free_symbols} but given {self.symbols}'
            )

        try:
            polys = {var: sp.Poly(self.expr, var) for var in self.expr.free_symbols}
            if any(poly.degree() > 1 for poly in polys.values()):
                raise sp.PolynomialError
        except sp.PolynomialError:
            raise ValueError(f'Expression is not linear')

        self.sym_coef_map = self.expr.as_coefficients_dict()
        self.linear_term: sp.Expr = sum([self.sym_coef_map[sym] * sym for sym in self.symbols if sym in self.expr.free_symbols])
        self.free_term = self.expr.subs({sym: 0 for sym in self.expr.free_symbols})

    def is_in_integer_shift(self) -> bool:
        """
        Checks if the hyperplane passes in shift points.
        :param shift: shift of each point in the lattice
        :return: True if passes in shift points. False otherwise.
        """
        coeffs = list(self.sym_coef_map.values())
        if (gcd := sp.gcd(coeffs) if coeffs else 0) == 0:
            return self.free_term == 0
        return sp.Abs(self.free_term) % gcd == 0

    def apply_shift(self, shift: Position) -> 'Hyperplane':
        """
        Applies a shift to this hyperplane.
        :param shift: a shift in each axis
        :return: The shifted hyperplane
        """
        # TODO: I think this is wrong, maybe should be a minus instead of a plus here????
        expr = self.expr.subs({sym: sym - shift[sym] for sym in self.expr.free_symbols})
        return Hyperplane(expr, symbols=self.symbols)

    @cached_property
    def equation_like(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        :return: lhs = rhs where lhs is the linear term and the rhs is the free trem
        """
        return self.linear_term, self.free_term

    @cached_property
    def vectors(self):
        linear = [self.sym_coef_map.get(sym, 0) for sym in self.symbols]
        return np.array(linear), self.free_term

    @property
    def as_below_vector(self):
        """
        linear + free <= 0 ---> linear <= -free
        """
        linear, free = self.vectors
        return linear, -free

    @property
    def as_above_vector(self):
        """
        linear + free >= 0 ---> linear >= -free --> -linear <= free
        """
        linear, free = self.vectors
        return -linear, free

    def __eq__(self, other: "Hyperplane"):
        if len(self.symbols) != len(other.symbols):
            return False
        if self.symbols == other.symbols:   # If symbols in the same order do simple equality
            linear, free = self.equation_like
            linear2, free2 = other.equation_like
            return (linear == linear2 and free == free2) or (linear == -linear2 and free == -free2)

        # If symbols are ordered in another way still check for equality
        neg = False
        first = True
        for sym in self.symbols:
            if sym not in other.symbols:
                return False
            if self.sym_coef_map[sym] == -other.sym_coef_map[sym]:
                if first:
                    neg = True
                elif not neg:
                    return False
            elif self.sym_coef_map[sym] == other.sym_coef_map[sym]:
                if neg:
                    return False
            else:
                return False
            first = False

        if neg:
            return self.free_term == -other.free_term
        return self.free_term == other.free_term

    def __hash__(self):
        return hash((self.expr, frozenset(self.symbols)))


if __name__ == '__main__':
    x, y, z, a = sp.symbols('x y z a')
    expr = 2*x+4*z-2*y+5
    hp = Hyperplane(expr, [a, x, y, z])

    print(hp.equation_like)
    print(hp.vectors)
