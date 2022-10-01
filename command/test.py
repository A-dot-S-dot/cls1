"""This module provides a task for performing unit tests.

"""
import subprocess

from .command import Command


class TestCommand(Command):
    def execute(self):
        subprocess.call(["test/test"] + self._args.file)
