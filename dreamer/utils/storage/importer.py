import json
import os
import pickle as pkl
from .formats import *


class Importer:
	"""
    A utility class for importing data from pickle or JSON files.
    """
    
    @classmethod
    def imprt(cls, path: str):
        """
        Imports data from path
        :param path: Path to the file where the data is stored.
        """
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
        """
        A generatorr for data (imports data from direcotry in chunks)
        :param path: Path of direcotry to import from as stream
        """
        
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory")

        for f in os.listdir(path):
            yield cls.imprt(os.path.join(path, f))
