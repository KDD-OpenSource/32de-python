import numpy as np
from typing import List
from util.datastructures import MetaPath
from .hypothesis import Hypothesis
from abc import ABC, abstractmethod

class AbstractMetaPathSelector(ABC):
    """
        Abstract base class for all objects that deliver meta paths based on an hypothesis.
    """
    @abstractmethod
    def get_next(self, size:int) -> List[MetaPath]:
        """
        :param size: the number of meta paths that are delivered.
        :return: A size-long list of meta paths.
        """
        pass

    @abstractmethod
    def get_all(self) -> List[MetaPath]:
        """
        :return: A list of all meta paths.
        """
        pass

    @abstractmethod
    def get_all_unrated(self) -> List[MetaPath]:
        """
        :return: A list of all unrated meta paths.
        """
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
        A meta path selector that selects meta paths randomly.
    """

    VISITED = 0
    NOT_VISITED = 1

    def __init__(self, meta_paths: List[MetaPath], hypothesis: Hypothesis=None):
        self.meta_paths = np.array(meta_paths)
        self.visited = np.array([self.NOT_VISITED] * len(meta_paths))
        self.hypothesis = hypothesis

    def get_next(self, size=1) -> (List[MetaPath], bool):
        """
        :return: requested number of next meta-paths to be shown.
        """
        last_batch = False
        if sum(self.visited) < size:
            size = sum(self.visited)
            last_batch = True

        idx = np.random.choice(range(len(self.meta_paths)), replace=False, p=self._prob_choose_meta_path(), size=size)
        self.visited[idx] = self.VISITED
        return self.meta_paths[idx], last_batch

    def get_all(self) -> List[MetaPath]:
        return self.meta_paths

    def get_all_unrated(self) -> List[MetaPath]:
        return [mp for i, mp in enumerate(self.meta_paths) if self.visited[i] == self.NOT_VISITED]

    """
    Functions
    """
    def _prob_choose_meta_path(self) -> np.ndarray:
        return self.visited / sum(self.visited) if sum(self.visited) else np.append(np.zeros(len(self.meta_paths)-1), [1])
