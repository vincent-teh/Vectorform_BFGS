from abc import ABC, abstractmethod
import json
import os
from typing import Callable


def generate_incremental_filename(
        filepath: str, filename: str, extension: str, start_point: int = 0
        ) -> str:
    """
    Args:
        filepath (str): Directory which file is stored.
        filename (str): Name of the file.
        extension (str): E.g. '.json', '.txt', etc.
        start_point (int, optional): Starting point of incremental. Defaults to 0.

    Returns:
        str: updated incremented file name

    Desc:
        Checks if filename-0 exists, if yes continue increment until suitable
        filename with correct numbers was found.
    """
    filename = os.path.join(filepath, filename)
    i = start_point
    _filename = filename + '-' + str(i) + extension
    while os.path.exists(_filename):
        i += 1
        _filename = filename + '-' + str(i) + extension
    return _filename


class FileSaver(ABC):
    @abstractmethod
    def save(self, data, filepath: str, filename: str):
        """Save the files based on input data"""


class JsonFileSaver(FileSaver):
    """
    Generate .json format files for given wihthin given directory.

    Args:
        filename_generator_fn (Callable[[str, str, str], str]): Generates the desired filename based on custom.
    """
    def __init__(self, filename_generator_fn: Callable[[str, str, str], str] = generate_incremental_filename) -> None:
        self.filename_generator_fn = filename_generator_fn

    def create_filepath(self, filepath: str):
        parent_dir, _ = os.path.split(filepath)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        if not os.path.exists(filepath):
            os.mkdir(filepath)

    def save(self, data, filepath: str, filename: str):
        self.create_filepath(filepath)
        filename = self.filename_generator_fn(filepath, filename, '.json')
        with open(filename, 'w') as file:
            json.dump(data, file)
