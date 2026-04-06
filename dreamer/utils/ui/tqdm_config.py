from tqdm import tqdm
from dreamer.utils.logger import Logger


class SmartTQDM(tqdm):
    _depth = 0

    def __init__(self, *args, **kwargs):
        is_top_level = (SmartTQDM._depth == 0)
        SmartTQDM._depth += 1
        if 'leave' not in kwargs:
            kwargs.update({'leave': is_top_level})

        # Determine the step size for 10% jumps if 'total' is known
        self.min_log_step = 0
        if 'total' in kwargs and kwargs['total']:
            self.min_log_step = max(1, kwargs['total'] // 10)
        elif len(args) > 0 and hasattr(args[0], '__len__'):
            self.min_log_step = max(1, len(args[0]) // 10)

        super().__init__(*args, **kwargs)
        self.last_print_func = Logger.print_func
        Logger.print_func = self.write
        self._last_logged_n = 0

    def update(self, n=1):
        """
        Overrides update to force a flush every 10%.
        """
        displayed = super().update(n)

        # Check if we crossed the next 10% threshold
        if self.min_log_step > 0:
            if (self.n - self._last_logged_n) >= self.min_log_step:
                percent = int(100 * self.n / self.total)
                Logger(
                    f"System Progress: {self.n} / {self.total} ({percent}%)",
                    Logger.Levels.debug
                ).log()
                self._last_logged_n = self.n
        return displayed

    def close(self):
        SmartTQDM._depth -= 1
        Logger.print_func = self.last_print_func
        super().close()
