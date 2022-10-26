import subprocess
from typing import List

from .command import Command


class Test(Command):
    _file: List[str]

    def __init__(self, file=None):
        self._file = file or []

    def execute(self):
        subprocess.call(["test/test"] + self._file)
