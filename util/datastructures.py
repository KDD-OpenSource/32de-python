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
            nodes = ['(n{}:{})'.format(i, node) for i, node in enumerate(self._nodes)]
            edges = ['[e{}:{}]'.format(i, edge) for i, edge in enumerate(self._edges)]
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
