from typing import List


class MetaPath:
    _edges = None
    _nodes = None

    def __init__(self, nodes: List[str] = [], edges: List[str] = []):
        assert (len(nodes) - 1 == len(edges)) or (len(nodes) == 0 and len(
            edges) == 0), "Invalid path: number of edges and nodes do not match."
        self._edges = edges
        self._nodes = nodes

    def is_empty(self) -> bool:
        return len(self) == 0

    def as_list(self) -> List[str]:
        representation = [None] * len(self)
        representation[::2] = self._nodes
        representation[1::2] = self._edges
        return representation

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
