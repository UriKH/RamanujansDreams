import sympy as sp
from typing import Union, List, Tuple, Dict, Set, Any, Generator, FrozenSet, Optional, Type, TextIO, Callable
from ramanujantools.cmf import CMF, pFq
from ramanujantools import Position
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShiftCMF:
    cmf: CMF
    shift: Position
    selected_points: Optional[List[Tuple[int | sp.Rational]]] = None
    only_selected: bool = False
    raw: bool = False


Shift = Union[sp.Rational | int | None]     # a shift in starting point
CMFtup = Tuple[CMF, Position]               # CMF tuple (CMF, list of shifts)
CMFlist = List[CMFtup]
EqTup = Tuple[sp.Expr, sp.Expr]             # Hyperplane equation representation

ShardVec = Tuple[int, ...]


