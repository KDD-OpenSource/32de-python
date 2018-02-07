import numpy as np
from typing import List
from util.datastructures import MetaPath
from .hypothesis import Hypothesis
from abc import ABC, abstractmethod

class AbstractMetaPathSelector(ABC):

    @abstractmethod
    def get_next(self, size:int) -> List[MetaPath]:
        pass

    @abstractmethod
    def get_all(self) -> List[MetaPath]:
        pass

    @abstractmethod
    def get_all_unrated(self) -> List[MetaPath]:
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractMetaPathSelector:
            mandatory_methods = ["get_all", "get_next", "get_all_unrated"]
            if all(any(method in B.__dict__ for B in C.__mro__) for method in mandatory_methods):
                return True
        return NotImplemented

class RandomMetaPathSelector(AbstractMetaPathSelector):
    """
    A MetaPathSelector selects the next meta-path to be rated.
    """

    VISITED = 0
    NOT_VISITED = 1

    def __init__(self, meta_paths: List[MetaPath], hypothesis: Hypothesis=None):
        self.meta_paths = np.array(meta_paths)
        self.visited = np.array([self.NOT_VISITED] * len(meta_paths))
        self.hypothesis = hypothesis

    def get_next(self, size=1) -> List[MetaPath]:
        """
        :return: requested number of next meta-paths to be shown.
        """
        idx = np.random.choice(range(len(self.meta_paths)), replace=False, p=self._prob_choose_mp(), size=size)
        self.visited[idx] = self.VISITED
        return self.meta_paths[idx]

    def get_all(self) -> List[MetaPath]:
        return self.meta_paths

    def get_all_unrated(self) -> List[MetaPath]:
        return [mp for i, mp in enumerate(self.meta_paths) if self.visited[i] == self.NOT_VISITED]

    """
    Functions
    """
    def _prob_choose_mp(self) -> np.ndarray:
        return self.visited / sum(self.visited)
