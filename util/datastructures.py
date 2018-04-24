from typing import List


class MetaPath:
    def __init__(self, **kwargs):
        """
        Create a Metapaths either from
        - a list of 'nodes' and 'edges' (ordered)
        - an 'edge_node_list'
        If no argument is given, an empty Metapath is created.
        """
        self._edges = None
        self._nodes = None
        self._embedding = None
        self._structural_value = None
        if 'nodes' in kwargs.keys() and 'edges' in kwargs.keys():
            nodes = kwargs['nodes']
            edges = kwargs['edges']
            assert (len(nodes) - 1 == len(edges)) or (
                    len(nodes) == 0 and len(edges) == 0), "Invalid path: number of edges and nodes do not match."
            self._edges = edges
            self._nodes = nodes

        elif 'edge_node_list' in kwargs.keys():
            edge_node_list = kwargs['edge_node_list']
            self._nodes = edge_node_list[::2]
            self._edges = edge_node_list[1::2]

        elif len(kwargs) == 0:
            self._nodes = []
            self._edges = []
        else:
            raise ValueError("Keywords not  valid: {}".format(', '.join(kwargs.keys())))

    def store_embedding(self, embedding: List[float]):
        self._embedding = embedding
        return self

    def store_structural_value(self, structural_value: float):
        self._structural_value = structural_value
        return self

    def is_empty(self) -> bool:
        return len(self) == 0

    def transform_representation(self, node_map, label_map):
        self._edges = [label_map[edge.encode()].decode() for edge in self._edges]
        self._nodes = [node_map[node.encode()].decode() for node in self._nodes]
        return self

    def as_list(self) -> List[str]:
        representation = [None] * len(self)
        representation[::2] = self._nodes
        representation[1::2] = self._edges
        return representation

    def get_structural_value(self) -> float:
        return self._structural_value

    def __copy__(self):
        return type(self)(nodes=self._nodes, edges=self._edges)

    def __len__(self) -> int:
        return len(self._edges) + len(self._nodes)

    def __str__(self) -> str:
        return ' '.join(map(str, self.as_list()))

    def number_node_types(self):
        return len(self._nodes)

    def get_representation(self, representation_type='UI'):
        if representation_type == 'UI':
            return self.as_list()
        elif representation_type == 'embedding':
            return self._embedding
        elif representation_type == 'query':
            nodes = ['[n{}:{}]'.format(i, node) for i, node in enumerate(self._nodes)]
            edges = ['(e{}:{})'.format(i, edge) for i, edge in enumerate(self._edges)]
            mp_query = [None] * len(self)
            mp_query[::2] = nodes
            mp_query[1::2] = edges
            return '-'.join(mp_query)
        else:
            raise ValueError("Representation type is not available {}".format(representation_type))


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

        # class MetaPathRatingGraph:
        #
        #     def __init__(self,rating=None):
        #         self.graph = Graph(directed=True)
        #         self.distance = self.graph.new_edge_property("double")
        #         self.meta_path_to_vertex = {}
        #         self.vertex_to_meta_path = {}
        #         if rating is not None:
        #             ordered = sorted(rating.items(), key=lambda tuple: tuple[1])
        #             list(map(lambda rated_mp_1, rated_mp_2: self.add_user_rating(rated_mp_1[0], rated_mp_2[0],
        #                                                                                 rated_mp_2[1] - rated_mp_1[1]),
        #                      ordered[:-1], ordered[1:]))
        #
        #     def __add_meta_path(self, meta_path: MetaPath) -> Vertex:
        #         """
        #         :param meta_path: The index of the meta-path will be retrieved or the meta-path is mapped to a new index.
        #         :return: Return the vertex corresponding to the meta-path
        #         """
        #         if meta_path in self.meta_path_to_vertex:
        #             meta_path_vertex = self.meta_path_to_vertex[meta_path]
        #         else:
        #             meta_path_vertex = self.graph.add_vertex()
        #             self.meta_path_to_vertex[meta_path] = meta_path_vertex
        #             self.vertex_to_meta_path[meta_path_vertex] = meta_path
        #         return meta_path_vertex
        #
        #     def add_user_rating(self, superior_meta_path: MetaPath, inferior_meta_path: MetaPath, distance: float) -> None:
        #         """
        #            :param superior_meta_path: The meta-path, which was rated higher.
        #            :param inferior_meta_path: The meta-path, which was rated lower.
        #            :param distance: The distance between meta-paths a and b.
        #         """
        #         assert (distance >= 0), "Distance may not be negative"
        #         superior_meta_path_vertex = self.__add_meta_path(superior_meta_path)
        #         inferior_meta_path_vertex = self.__add_meta_path(inferior_meta_path)
        #
        #         new_edge_positive = self.graph.add_edge(superior_meta_path_vertex, inferior_meta_path_vertex)
        #         self.distance[new_edge_positive] = distance
        #
        #     def number_of_edges(self) -> int:
        #         return self.graph.num_edges()
        #
        #     def number_of_nodes(self) -> int:
        #         return self.graph.num_vertices()
        #
        #     def all_nodes(self) -> List[MetaPath]:
        #         return list(self.meta_path_to_vertex.keys())
        #
        #     def all_pair_distances(self) -> PropertyMap:
        #         """
        #         :return: A Property Map with raw all-pair distances in it.
        #         """
        #         return shortest_distance(self.graph, weights=self.distance, directed=True)
        #
        #     def stream_meta_path_distances(self) -> (MetaPath, MetaPath, float):
        #         """
        #         :return: A stream of MetaPath-Distance-Triple.
        #         """
        #         dist_map = self.graph.new_vp("double", numpy.inf)
        #         for node in self.graph.vertices():
        #             distances_from_source = shortest_distance(self.graph, source=node, weights=self.distance, directed=True, dist_map=dist_map)
        #             for target_index, distance in enumerate(distances_from_source):
        #                 if not numpy.isinf(distance) and distance != 0:
        #                     yield self.vertex_to_meta_path[node], self.vertex_to_meta_path[self.graph.vertex(target_index)], distance
        #
        #     def __str__(self):
        #         return "MetaPathRating with {} rating(s)".format(len(self.graph.get_edges()))
        #
        #     def draw(self, filename='log/rating.png'):
        #         layout = arf_layout(self.graph, max_iter=0)
        #         # TODO: maybe add weights
        #         graph_draw(self.graph, pos=layout, vertex_fill_color=[0, 0, 1.0, 1.0],
        #                    output=filename)  # , edge_pen_width=pen_width)
        #         print('Printed rating to file {}'.format(filename))
