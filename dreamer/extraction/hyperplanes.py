from ramanujantools import Position
from dataclasses import dataclass
from dreamer.utils.caching import cached_property

import sympy as sp
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class Hyperplane:
    """
    Represents a hyperplane as a sympy expression.
    The expression might miss some symbols as the space the hyperplane lives in consists of more axes than defined.
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
            raise ValueError(f'Expression is not linear: {self.expr}')

        coef_dict = self.expr.as_coefficients_dict()
        self.sym_coef_map = {k: coef_dict[k] if k in coef_dict.keys() else 0 for k in self.symbols + [1]}
        self.free_term = self.sym_coef_map.get(1, 0)
        self.linear_term = self.expr - self.free_term

        for sym in self.symbols:
            if self.sym_coef_map[sym] == 0:
                continue
            if self.sym_coef_map[sym] < 0:
                self.linear_term = -self.linear_term
                self.free_term = -self.free_term
                self.expr = -self.expr
            break

    def is_in_integer_shift(self) -> bool:
        """
        Checks if the hyperplane passes in shift points (Integer Solutions check).
        """
        coeffs = [self.sym_coef_map.get(s, sp.Integer(0)) for s in self.symbols]

        # if trivial c = 0
        if all(c == 0 for c in coeffs):
            return self.free_term == 0

        all_terms = coeffs + [self.free_term]
        common_denom = sp.Integer(1)
        for val in all_terms:
            common_denom = sp.lcm(common_denom, sp.denom(val))

        int_coeffs = [sp.Integer(c * common_denom) for c in coeffs]
        int_free_term = sp.Integer(self.free_term * common_denom)
        coeffs_gcd = sp.gcd(int_coeffs)
        return int_free_term % coeffs_gcd == 0

    def apply_shift(self, shift: Position) -> 'Hyperplane':
        """
        Applies a shift to this hyperplane.
        :param shift: a shift in each axis
        :return: The shifted hyperplane
        """
        expr = self.expr.subs({sym: sym + shift[sym] for sym in self.expr.free_symbols})
        return Hyperplane(expr, symbols=self.symbols)

    # def remove_shift(self, shift: Position) -> 'Hyperplane':
    #     """
    #     Applies a shift to this hyperplane.
    #     :param shift: a shift in each axis
    #     :return: The shifted hyperplane
    #     """
    #     expr = self.expr.subs({sym: sym - shift[sym] for sym in self.expr.free_symbols})
    #     return Hyperplane(expr, symbols=self.symbols)

    @cached_property
    def equation_like(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        :return: lhs + rhs = 0 where lhs is the linear term and the rhs is the free trem
        """
        return self.linear_term, self.free_term

    @cached_property
    def vectors(self) -> Tuple[np.ndarray, float]:
        """
        :return: a vector representation of the hyperplane (linear term as a coefficient vector, free term)
        """
        linear = [self.sym_coef_map.get(sym, 0) for sym in self.symbols]
        return np.array(linear), self.sym_coef_map.get(1, 0)

    @property
    def as_below_vector(self) -> Tuple[np.ndarray, float]:
        """
        :return: A vector representation of the hyperplane in the form (linear term, -free term)

        [ linear + free <= 0 ---> linear <= -free ]
        """
        linear, free = self.vectors
        return linear, -free

    @property
    def as_above_vector(self) -> Tuple[np.ndarray, float]:
        """
        :return: A vector representation of the hyperplane in the form (-linear term, free term)

        linear + free >= 0 ---> linear >= -free --> -linear <= free
        """
        linear, free = self.vectors
        return -linear, free

    def __eq__(self, other: "Hyperplane"):
        if self.symbols != other.symbols:
            return False
        linear, free = self.vectors
        linear2, free2 = other.vectors
        return (np.array_equal(linear, linear2) and (free == free2)) or (np.array_equal(linear, -linear2) and (free == -free2))

    def __hash__(self):
        return hash((self.equation_like, frozenset(self.symbols)))


if __name__ == '__main__':
    x, y, z, a = sp.symbols('x y z a')
    x0, x1, y0 = sp.symbols('x0 x1 y0')
    expr = 2*x+4*z-2*y+5
    hp = Hyperplane(expr, [a, x, y, z])
    hp1 = Hyperplane(-x0 + y0, [x0, x1, y0])
    hp2 = Hyperplane(x0 - y0, [x0, x1, y0])
    print(hp1 == hp2)
    # print(hp.equation_like)
    # print(hp.vectors)
