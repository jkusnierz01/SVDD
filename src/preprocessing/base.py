from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    @abstractmethod
    def preprocess(self) -> dict:
        pass