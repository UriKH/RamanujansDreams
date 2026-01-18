import functools
import time
import inspect
from enum import Enum, auto
from contextlib import contextmanager
from dreamer.configs import logging_config
from typing import Callable

from typing import Dict, Tuple


class Logger:
    """
    Logging for terminal interface and debugging
    """
    print_func: Callable = print
    timer_mapping: Dict[str, Tuple[int, float]] = dict()

    class Levels(Enum):
        message = auto()
        info = auto()
        warning = auto()
        inform = auto()
        exception = auto()
        fatal = auto()

    class Colors:
        white = '\033[0m'
        red = '\033[91m'
        green = '\033[92m'
        yellow = '\033[93m'

    def __init__(self, msg, level=Levels.message, end_with_nl=True, condition=True):
        calling_frame = inspect.stack()[1]
        self.calling_function_name = calling_frame.function
        self.level = level if isinstance(level, Logger.Levels) else Logger.Levels.info
        self.msg = msg
        self.condition = condition
        if end_with_nl:
            self.end = f'{Logger.Colors.white}\n'
        else:
            self.end = f'{Logger.Colors.white} '

    def time_it(self, func):
        """
        Decorator to log the processing time of a function
        :param func: the function to time.
        :return: the wrapped function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwarg):
            start = time.time()
            result = func(*args, **kwarg)
            end = time.time()

            if round(end - start, 3) != 0 and logging_config.PROFILE:
                Logger(
                    self.msg + f'Call to "{func.__name__}" finished in {(end - start):.3f} seconds',
                    self.level
                ).log(msg_prefix='\n')
            return result
        return wrapper

    def log(self, msg_prefix='', in_function: bool = False):
        """
        Log a message with it's logging level to the standard output
        :param msg_prefix: the message level prefix for printing
        :param in_function: Add the calling function name to the message
        """
        if not self.condition:
            return

        match self.level:
            case Logger.Levels.message:  # message
                self.print_func(f'{Logger.Colors.white}{msg_prefix}{self.msg}', end=self.end)
            case Logger.Levels.info:       # info - green
                self.print_func(f'{Logger.Colors.green}{msg_prefix}[INFO] {self.msg}', end=self.end)
            case Logger.Levels.warning:    # warning - yellow
                self.print_func(f'{Logger.Colors.yellow}{msg_prefix}[WARNING] {self.msg}', end=self.end)
            case Logger.Levels.inform:     # does not raise exception - red
                msg = f'{Logger.Colors.red}{msg_prefix}[NOTICE] {self.msg}'
                if in_function:
                    msg += f' in {self.calling_function_name}'
                self.print_func(msg, end=self.end)
            case Logger.Levels.exception:  # exception - red
                msg = f'{Logger.Colors.red}{msg_prefix}[ERROR] {self.msg}'
                if in_function:
                    msg += f' in {self.calling_function_name}'
                self.print_func(msg, end=self.end)
            case Logger.Levels.fatal:      # fatal error - red
                self.print_func(f'{Logger.Colors.red}{msg_prefix}[ERROR] {self.msg} in {self.calling_function_name} \n\t'
                      f'-> exiting', end=self.end)
                try:
                    __IPYTHON__
                    raise Exception('Stopping...')
                except NameError:
                    exit(1)
        return

    @staticmethod
    def log_exec(func):
        """
        :param func: A function to run.
        :return: the wrapped function
        """
        if not callable(func):
            raise Exception('Function is not callable, bad decorator')

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            Logger(f'Executing {func.__name__}', Logger.Levels.info).log()
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            Logger(f'Finished execution {func.__name__} in {end - start} seconds', Logger.Levels.info).log()
            return res
        return wrapper

    @staticmethod
    def buffer_print(total: int, text: str, char: chr):
        """
        Format a message into a buffer
        :param total: The buffer size
        :param text: Text to be printed
        :param char: Buffer constructed by the char
        :return: The formatted buffer string
        """
        txt_len = len(text) + 2
        t = int((total - txt_len) / 2)
        if txt_len + 2 * t != 150:
            return f'{char * t} {text} {char * (t + 1)}'
        return f'{char * t} {text} {char * t}'

    @classmethod
    @contextmanager
    def simple_timer(cls, label):
        """
        A lightweight generator-based context manager.
        Useful for quick debugging without needing a class instance.
        :param label: A label for the timer.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            if logging_config.PROFILE:
                print(f"{label}: {end - start:.6f} seconds")
            if logging_config.PROFILE_SUMMARY:
                n, s = cls.timer_mapping.get(label, (0, 0.0))
                cls.timer_mapping[label] = (n + 1, s + (end - start))

    @classmethod
    def timer_summary(cls):
        if not logging_config.PROFILE_SUMMARY:
            return
        cls('\n======= profile summary ======').log()
        for label, (n, s) in cls.timer_mapping.items():
            cls(f"{label}: {n} runs, avg time: {s / n:.6f} seconds").log()

    @staticmethod
    def sleep(t: float) -> None:
        """
        Sleeps for a given time if configured to do so
        :param t: Time to sleep
        """
        if logging_config.SLEEP_TO_PRINT:
            time.sleep(t)
