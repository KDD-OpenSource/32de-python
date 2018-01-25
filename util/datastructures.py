from typing import List
from graph_tool.all import Graph, shortest_distance
import numpy

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


class MetaPathRatingGraph:

    def __init__(self):
        self.g = Graph(directed=True)
        self.distance = self.g.new_edge_property("double")
        self.meta_path_to_vertex = {}
        self.vertex_to_meta_path = {}


    def __add_meta_path(self, meta_path: MetaPath):
        """
        :param a: The index of the meta-path will be retrieved or the meta-path is mapped to a new index.
        """
        if meta_path in self.meta_path_to_vertex:
            meta_path_vertex = self.meta_path_to_vertex[meta_path]
        else:
            meta_path_vertex = self.g.add_vertex()
            self.meta_path_to_vertex[meta_path] = meta_path_vertex
            self.vertex_to_meta_path[meta_path_vertex] = meta_path
        return meta_path_vertex



    def add_user_rating(self, superior_meta_path: MetaPath, inferior_meta_path: MetaPath, distance: float):
        """
           :param a: The meta-path, which was rated higher compared to b.
           :param b: The meta-path, which was rated lower compared to a.
           :param distance: The distance between meta-paths a and b.
        """
        assert (distance >= 0), "Distance may not be negative"
        superior_meta_path_index = self.__add_meta_path(superior_meta_path)
        inferior_meta_path_index = self.__add_meta_path(inferior_meta_path)

        new_edge_positive = self.g.add_edge(superior_meta_path_index, inferior_meta_path_index)
        self.distance[new_edge_positive] = distance


    def number_of_edges(self) -> int:
        return self.g.num_edges()

    def number_of_nodes(self) -> int:
        return self.g.num_vertices()

    def all_nodes(self) -> List[MetaPath]:
        return list(self.meta_path_to_vertex.keys())

    def all_pair_distances(self):
        """
        :return: A Property Map with raw all-pair distances in it.
        """
        return shortest_distance(self.g, weights=self.distance, directed=True)

    def stream_meta_path_distances(self):
        """
        :return: A stream of MetaPath-Distance-Triple.
        """
        dist_map = self.g.new_vp("double", numpy.inf)
        for node in self.g.vertices():
            distances_from_source = shortest_distance(self.g, source=node, weights=self.distance, directed=True, dist_map=dist_map)
            for target_index, distance in enumerate(distances_from_source):
                if not numpy.isinf(distance) and distance != 0:
                    yield self.vertex_to_meta_path[node], self.vertex_to_meta_path[self.g.vertex(target_index)], distance

