import json
import os
import pickle as pkl
from .formats import *


class Importer:
    @classmethod
    def imprt(cls, path: str):
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        if os.path.isdir(path):
            data = dict()
            for f in os.listdir(path):
                data[f] = cls.imprt(os.path.join(path, f))
            return data

        match path.split('.')[-1]:
            case Formats.JSON.value:
                with open(path, 'r') as f:
                    return json.load(f)
            case Formats.PICKLE.value:
                with open(path, 'rb') as f:
                    return pkl.load(f)
            case _:
                raise ValueError(f"File {path} has unsupported format")

    @classmethod
    def import_stream(cls, path):
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory")

        for f in os.listdir(path):
            yield cls.imprt(os.path.join(path, f))
