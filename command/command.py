from abc import ABC, abstractmethod


class Command(ABC):
    @abstractmethod
    def execute(self):
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__
