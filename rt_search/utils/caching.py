from functools import lru_cache
import functools


def cached_property(func, ignore_pickle=True):
    prop = functools.cached_property(func)
    prop._ignore_pickle = ignore_pickle
    return prop

