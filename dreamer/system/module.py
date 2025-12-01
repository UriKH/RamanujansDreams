from abc import ABC, abstractmethod
from inspect import isclass
from typing import Optional
import functools
import traceback

from ..utils.logger import Logger


class CatchErrorInModule:
    """
    Wrapper class for catching errors in modules or other classes globally
    """

    def __init__(self, with_trace: bool = True, fatal: bool = False):
        """
        :param with_trace: Print also the error trace
        :param fatal: if True then exit else continue
        """
        self.with_trace = with_trace
        self.fatal = fatal

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(mod, *args, **kwargs):
            try:
                return func(mod, *args, **kwargs)
            except Exception as e:
                version = None
                if isinstance(mod, Module):
                    name = mod.name
                    version = mod.version
                else:
                    name = mod.__name__ if isclass(mod) else mod.__class__.__name__

                Logger(
                    f'In module {name} [version: {version}] caught {e.__class__.__name__}: {e}',
                    Logger.Levels.exception
                ).log()
                if self.with_trace:
                    traceback.print_exc()
                if self.fatal:
                    try:
                        # This will throw an exception instead of exit if you are using a Notbook
                        __IPYTHON__
                        raise Exception('Stopping...')
                    except NameError:
                        exit(1)
        return wrapper


class Module(ABC):
    """
    The basic building block of the system is this module
    """

    def __init__(self,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 version: Optional[str] = None):
        """
        :param name: Module name
        :param description: Description of the module
        :param version: A version of the module
        """
        self.name = name if name else self.__class__.__name__
        self.description = description
        self.version = version

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Executes the module. The return value and arguments are specified by the child classes
        """
        raise NotImplementedError
