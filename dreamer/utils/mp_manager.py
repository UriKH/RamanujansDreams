from concurrent.futures import ProcessPoolExecutor
from dreamer.configs import config


def init_worker(config_overrides):
    from dreamer.configs import config
    config.configure(**config_overrides)


def create_pool():
    return ProcessPoolExecutor(initializer=init_worker, initargs=(config.export_configurations(),))
