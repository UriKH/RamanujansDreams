from tqdm import tqdm
from dreamer.utils.logger import Logger


class SmartTQDM(tqdm):
    _depth = 0

    def __init__(self, *args, **kwargs):
        is_top_level = (SmartTQDM._depth == 0)
        SmartTQDM._depth += 1
        if 'leave' not in kwargs:
            kwargs.update({'leave': is_top_level})
        super().__init__(*args, **kwargs)
        self.last_print_func = Logger.print_func
        Logger.print_func = self.write

    def close(self):
        SmartTQDM._depth -= 1
        Logger.print_func = self.last_print_func
        super().close()
