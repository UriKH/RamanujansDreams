import json
from typing import TypeVar, Generic
from enum import Enum, auto
import os

from ..types import *
from .exports import Exportable

T = TypeVar('T', bound=Exportable)

TYPE_ANNOTATE = 'type'
OBJ_ANNOTATE = 'object'


class Formats(Enum):
    JSON = 'json'
    PICKLE = 'pickle'
    PARQUET = 'parquet'


class Exporter(Generic[T]):
    class ExportableFormat(Enum):
        OBJECT = auto()
        LST_OBJ = auto()
        D_STR_OBJ = auto()
        D_STR_LST_OBJ = auto()
        ERROR = auto()

    def __init__(self, path: str, t: Optional[Type[T]] = None):
        self.cls = t
        self.path = path

    def __export_type_check(self, data):
        def is_list_of(l, t):
            try:
                return all(isinstance(x, t) for x in data)
            except:
                return False

        t = self.ExportableFormat.ERROR
        match data:
            case list():
                if not is_list_of(data, self.cls):
                    raise ValueError(f"data list must be a list of {self.cls}")
                return self.ExportableFormat.LST_OBJ
            case dict():
                if all(isinstance(k, str) and isinstance(v, self.cls) for k, v in data.items()):
                    return self.ExportableFormat.D_STR_OBJ
                elif all(isinstance(k, str) and is_list_of(v, self.cls) for k, v in data.items()):
                    return self.ExportableFormat.D_STR_LST_OBJ
                else:
                    raise ValueError(f"data dict must be a dictionary of Dict[str, List[{self.cls}]]")
            case self.cls():
                return self.ExportableFormat.OBJECT
            case _:
                raise ValueError(f"data must be: {self.cls} or List[{self.cls}] or Dict[str, List[{self.cls}]]")

    def __call__(
            self,
            data: Union[List[T], Dict[str, List[T]], T],
            as_single_file: Tuple[bool, str] = (False, None),
            save_as: Formats = Formats.JSON,
            metadata: Optional[Union[Dict, List]] = None
    ) -> List[str]:
        if save_as not in Formats:
            raise ValueError(f"save_as must be one of {[f for f in Formats]}")

        d_type = self.__export_type_check(data)
        # match d_type:
        #                         case self.ExportableFormat.OBJECT:
        #                             pass
        #                         case self.ExportableFormat.LST_OBJ:
        #                             pass
        #                         case self.ExportableFormat.D_STR_OBJ:
        #                             pass
        #                         case self.ExportableFormat.D_STR_LST_OBJ:
        #                             pass
        os.makedirs(self.path, exist_ok=True)
        match save_as:
            case Formats.JSON:
                if as_single_file[0] or d_type is self.ExportableFormat.OBJECT:
                    # TODO: create a json exportaion of the data into the file name given or just from object
                    with open(as_single_file[1], "w") as f:
                        json.dump({TYPE_ANNOTATE: self.cls.__name__, OBJ_ANNOTATE: data.to_json_obj()}, f)

                    """
                    1. Create directory V
                    2. match case - convert object/s into json
                    3. create file and write
                    """
                    pass
                else:
                    # TODO: seperate to files based on seperator key if given, else seperate by object
                    pass
            case Formats.PICKLE:
                pass
            case Formats.PARQUET:
                pass

        results = []
        if isinstance(data, list):
            with open(self.path, 'a+') as f:
                for d in data:
                    results.append(d.export_(f))
            return results
        return [data.export_(self.path)]

    @classmethod
    def __class_getitem__(cls, item: Type[T]):
        return lambda path: cls(path, item)
