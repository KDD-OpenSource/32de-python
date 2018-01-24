from typing import List
from graph_tool.all import *


class MetaPath:
    edges = None
    nodes = None

    def __init__(self):
        self.edges = []
        self.nodes = []

    def __init__(self, nodes, edges):
        assert (len(nodes) - 1 == len(edges)) or (
        len(nodes) == 0 and len(edges) == 0), "Path is not valid, number of edges and nodes does not compse a path"
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
        return ';'.join(map(str, self.as_list()))


class MetaPathRating:
    meta_path = None
    structural_value = None
    domain_value = None

    def __init__(self, meta_path: MetaPath):
        self.meta_path = meta_path


class MetaPathRatingGraph:

    meta_paths_map = None
    distance = None
    g = None

    def __init__(self):
        self.g = Graph(directed=False)
        self.distance = self.g.new_edge_property("double")
        self.meta_paths_map = {}

    def add_meta_path(self, a: MetaPath) -> None:
        if a in self.meta_paths_map:
            v = self.meta_paths_map[a]
        else:
            v = self.g.add_vertex()
            self.meta_paths_map[a] = v
        return v

    def add_user_rating(self, a: MetaPath, b: MetaPath, distance: float):
        id_a = self.add_meta_path(a)
        id_b = self.add_meta_path(b)

        new_edge = self.g.add_edge(id_a, id_b)
        self.distance[new_edge] = distance

    def all_nodes(self):
        return self.meta_paths_map.values()