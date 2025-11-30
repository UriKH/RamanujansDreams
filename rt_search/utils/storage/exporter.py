import json
import os
import pickle as pkl
import pyarrow.parquet as pq
from ..types import *
from .formats import *
import shutil


class Exporter:
    @classmethod
    def export(
        cls,
        root: str,
        f_name: str = None,
        exists_ok: bool = True,
        clean_exists: bool = False,
        fmt: Formats = Formats.PICKLE,
        data: Any = None
    ):
        # make sure root directory is valid
        os.makedirs(root, exist_ok=True)

        # if as single file named f_name
        if f_name:
            # cleanup if needed
            path = os.path.join(root, f_name + '.' + fmt.value)
            if os.path.exists(path) and not exists_ok:
                raise FileExistsError(f"File {path} already exists")

            # write file
            match fmt:
                case Formats.PARQUET:
                    pq.write_table(data, path)
                case Formats.PICKLE:
                    with open(path, "wb") as f:
                        pkl.dump(data, f)
                case Formats.JSON:
                    with open(path, "w") as f:
                        json.dump(data, f)
        else:
            # cleanup if needed
            if os.path.isdir(root) and not exists_ok:
                raise FileExistsError(f"File {root} already exists")
            elif os.path.exists(root) and clean_exists:
                shutil.rmtree(root)
            elif not isinstance(data, dict | list):
                raise TypeError(f"data must be a dict or list, got {type(data)}")

            if isinstance(data, dict):
                for i, (k, v) in enumerate(data.items()):
                    cls.export(root, f'{k}_{i+1}', exists_ok=exists_ok, fmt=fmt, data=v)
            else:
                for i, v in enumerate(data):
                    cls.export(root, f'{i+1}', exists_ok=exists_ok, fmt=fmt, data=v)
