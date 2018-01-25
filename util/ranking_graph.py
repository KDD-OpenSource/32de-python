from typing import List, Generic, TypeVar

T = TypeVar('T')

class RankingGraph(Generic[T]):

    def __init__(self):
        """

        """

    def all_nodes(self) -> List[T]:
        """

        :return: A list of all nodes in the graph.
        """
        raise NotImplementedError
        return []

    def transitive_closures(self) -> List[List[T]]:
        """

        :return: A list containing all transitive closures, where each closure is a list starting with the node
                 of which the closure is built.
        """
        raise NotImplementedError
        return [[]]