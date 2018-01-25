class MetaPath:

    edges = None
    nodes = None
    structural_score = None
    domain_score = None

    def __init__(self):
        self.edges = []
        self.nodes = []

    def __init__(self, nodes, edges):
        assert (len(nodes) - 1 == len(edges)) or (len(nodes) == 0 and len(edges) == 0), "Path is not valid, number of edges and nodes does not compse a path"
        self.edges = edges
        self.nodes = nodes

    def __init__(self, nodes, edges, structural_score, domain_score):
        assert (len(nodes) - 1 == len(edges)) or (len(nodes) == 0 and len(edges) == 0), "Path is not valid, number of edges and nodes does not compse a path"
        assert (structural_score is not None and domain_score is not None), "Structural score and domain score have to be set"
        self.edges = edges
        self.nodes = nodes
        self.structural_score = structural_score
        self.domain_score = domain_score

    def is_empty(self):
        return len(self) == 0

    def as_list(self):
        representation = [None] * len(self)
        representation[::2] = self.nodes
        representation[1::2] = self.edges
        return representation

    def __len__(self):
        return len(self.edges) + len(self.nodes)

    def __str__(self):
        return ';'.join(map(str,self.as_list()))
