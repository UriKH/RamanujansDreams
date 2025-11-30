import sympy as sp
from typing import Union, List, Tuple, Dict, Set, Any, FrozenSet, Optional, Type, TextIO, Callable
from ramanujantools.cmf import CMF, pFq
from ramanujantools import Position
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShiftCMF:
    cmf: CMF
    shift: Position


Shift = Union[sp.Rational | int | None]     # a shift in starting point
CMFtup = Tuple[CMF, Position]               # CMF tuple (CMF, list of shifts)
CMFlist = List[CMFtup]
EqTup = Tuple[sp.Expr, sp.Expr]             # Hyperplane equation representation

ShardVec = Tuple[int, ...]


