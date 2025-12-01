from typing import List
from dataclasses import fields, is_dataclass


class Configurable:
    def get_configurations(self) -> List[str]:
        if is_dataclass(self):
            return [f.name for f in fields(self)]
        return []

    def __repr__(self) -> str:
        configs = self.get_configurations()
        return f'{self.__class__.__name__}:{"\n* ".join(configs)}'
