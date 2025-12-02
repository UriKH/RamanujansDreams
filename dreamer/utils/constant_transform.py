import sympy as sp
import mpmath as mp
from dreamer.configs import (
    sys_config
)
from dreamer.system.errors import *
from typing import List


def get_const_as_mpf(constant: str) -> mp.mpf:
    """
    Convert string to a mpmath.mpf value
    :param constant: Constant name as string
    :raise UnknownConstant if constant is unknown
    :return: the mp.mpf value
    """
    try:
        constant = sys_config.SYMPY_TO_MPMATH[constant]
    except Exception:
        raise UnknownConstant(constant + UnknownConstant.default_msg)

    pieces = constant.split("-")
    if len(pieces) == 1:
        try:
            return getattr(sp, constant)
        except Exception:
            raise UnknownConstant(constant + UnknownConstant.default_msg)

    n = int(pieces[1])
    try:
        return getattr(sp, pieces[0])(n)
    except Exception:
        raise UnknownConstant(constant + UnknownConstant.default_msg)


def get_const_as_sp(constant: str):
    """
    Convert string to a sympy known value
    :param constant: Constant name as string
    :raise UnknownConstant if constant is unknown
    :return: the sympy constant
    """
    pieces = constant.split("-")
    if len(pieces) == 1:
        try:
            return getattr(sp, constant)
        except Exception:
            raise UnknownConstant(constant + UnknownConstant.default_msg)

    n = int(pieces[1])
    try:
        return getattr(sp, pieces[0])(n)
    except Exception:
        raise UnknownConstant(constant + UnknownConstant.default_msg)


def get_constants(constants: List[str] | str):
    """
    Retrieve the constants as sympy constants from strings
    :param constants: A list of constant names
    :return: The sympy constants
    """
    if isinstance(constants, str):
        constants = [constants]
    return {c: get_const_as_sp(c) for c in constants}
