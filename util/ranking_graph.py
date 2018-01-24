from typing import List

class RankingGraph:

    def __init__(self):
        """

        """
        raise NotImplementedError

    def all_nodes(self) -> List:
        """

        :return: A list of all nodes in the graph.
        """
        raise NotImplementedError
        return []

    def transitive_closure(self) -> List:
        """

        :return: A list containing all transitive closures, where each closure is a list starting with the node
                 of which the closure is built.
        """
        raise NotImplementedError
        return [[]]