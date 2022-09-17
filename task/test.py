"""This module provides a task for performing unit tests.

"""
from .task import Task
import subprocess


class TestTask(Task):
    def execute(self):
        subprocess.call(["test/test"] + self._args.file)
