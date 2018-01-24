from typing import List


class MetaPath:
    edges = None
    nodes = None

    def __init__(self):
        self.edges = []
        self.nodes = []

    def __init__(self, nodes, edges):
        assert (len(nodes) - 1 == len(edges)) or (len(nodes) == 0 and len(edges) == 0), "Path is not valid, number of edges and nodes does not compse a path"
        self.edges = edges
        self.nodes = nodes

    def is_empty(self) -> bool:
        return len(self) == 0

    def as_list(self) -> List:
        representation = [None] * len(self)
        representation[::2] = self.nodes
        representation[1::2] = self.edges
        return representation

    def __len__(self) -> int:
        return len(self.edges) + len(self.nodes)

    def __str__(self) -> str:
        return ';'.join(map(str,self.as_list()))


class MetaPathRating:
    meta_path = None
    structural_value = None
    domain_value = None

    def __init__(self, meta_path: MetaPath):
        self.meta_path = meta_path
