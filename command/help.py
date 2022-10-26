"""This module provides a task for displaying help messages.

"""
from argparse import ArgumentParser

from .command import Command


class Help(Command):
    parser: ArgumentParser

    def __init__(self, parser: ArgumentParser):
        self.parser = parser

    def execute(self):
        self.parser.print_help()
