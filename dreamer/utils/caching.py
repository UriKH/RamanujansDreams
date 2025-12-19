import functools


def cached_property(func, ignore_pickle=False):
    prop = functools.cached_property(func)
    prop._ignore_pickle = ignore_pickle
    return prop

