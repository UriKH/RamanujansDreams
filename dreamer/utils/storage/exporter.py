import json
import os
import pickle as pkl
from ..types import *
from .formats import *
import shutil
from contextlib import contextmanager


class Exporter:
    """
    A utility class for exporting data into files as pickle or JSON
    """
    
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
    	"""
        Exports data into a into a file in a specific format. 
        Note that there is no checking that the data could be written in this format.
        :param root: Direcotry in which to create the new file.
        :param f_name: Exported data file name.
        :param exists_ok: If false throws an error if the directory already exists.
        :param clean_exists: Removes all files from the direcotry if it already exists.
        :param fmt: Chosen format to export.
        :param data: Data to export.
        """
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
            elif isinstance(data, list):
                for i, v in enumerate(data):
                    cls.export(root, f'{i+1}', exists_ok=exists_ok, fmt=fmt, data=v)

    @classmethod
    @contextmanager
    def export_stream(
            cls,
            root: str,
            exists_ok: bool = True,
            clean_exists: bool = False,
            fmt: Formats = Formats.PICKLE
    ) -> Generator[Callable[[Any], None], None, None]:
        """
        Context manager for exporting a stream of data chunks.
        Yields a function that accepts data and writes it to a new file in the sequence.
        :param root: Directory to store the streamed data in.
        :param clean_exists: If true, removes all files from the already existing directory.
        :param fmt: Format to save the data in.
        """
        # Prepare the directory
        if os.path.exists(root):
            if clean_exists:
                shutil.rmtree(root)
                os.makedirs(root)
            elif not exists_ok:
                # Check if directory is empty, otherwise raise error if exists_ok is False
                if os.listdir(root):
                    raise FileExistsError(f"Directory {root} already exists and is not empty")
        else:
            os.makedirs(root)

        chunk_id = 0

        def write_chunk(data: Any) -> None:
            nonlocal chunk_id
            # Use zero-padding (e.g., chunk_0000000001) to ensure
            # file system sorting matches logical order.
            filename = f"chunk_{chunk_id:010d}"

            cls.export(
                root=root,
                f_name=filename,
                exists_ok=True,  # We control the names, so we assume we are safe to write
                fmt=fmt,
                data=data
            )
            chunk_id += 1

        try:
            yield write_chunk
        finally:
            pass
