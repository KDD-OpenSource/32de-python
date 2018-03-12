from typing import List
from graph_tool.all import *
import numpy


class MetaPath:
    _edges = None
    _nodes = None

    def __init__(self, nodes: List[str] = [], edges: List[str] = []):
        assert (len(nodes) - 1 == len(edges)) or (
            len(nodes) == 0 and len(edges) == 0), "Invalid path: number of edges and nodes do not match."
        self._edges = edges
        self._nodes = nodes

    def is_empty(self) -> bool:
        return len(self) == 0

    def as_list(self) -> List[str]:
        representation = [None] * len(self)
        representation[::2] = self._nodes
        representation[1::2] = self._edges
        return representation


    @staticmethod
    def from_list(meta_path: List):
        assert len(meta_path) % 2 != 0 or len(meta_path) == 0
        return MetaPath(meta_path[::2], meta_path[1::2])

    @staticmethod
    def from_string(meta_path: str, sep=' '):
        return MetaPath.from_list(meta_path.split(sep))

    def __len__(self) -> int:
        return len(self._edges) + len(self._nodes)

    def __str__(self) -> str:
        return ' '.join(map(str, self.as_list()))


class MetaPathRating:
    # TODO: Define methods
    meta_path = None
    structural_value = None
    domain_value = None

    def __init__(self, meta_path: MetaPath):
        self.meta_path = meta_path

    def __init__(self, meta_path: MetaPath, structural_value: float, domain_value: float):
        self.meta_path = meta_path
        self.structural_value = structural_value
        self.domain_value = domain_value


class UserOrderedMetaPaths:
    meta_paths = None
    distances = None

    def __init__(self, meta_paths: List[MetaPath], distances: List[float] = None):
        """

        :param distances: Distances between meta-paths on UI
        :param meta_paths: All meta-paths as ordered by the user
        which were rated in one "session" (between each click on "Next five")
        """
        self.meta_paths = meta_paths
        self.set_distances(distances)

    def set_distances(self, distances: List[float]) -> None:
        assert distances is None or len(self.meta_paths) == len(
            distances), 'Number of meta-paths which is {} doesn\'t match number of passed distances which is {}'.format(
            len(self.meta_paths), len(distances))
        self.distances = distances

