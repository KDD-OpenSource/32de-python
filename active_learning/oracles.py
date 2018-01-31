from util.datastructures import MetaPath
from typing import Dict


class Oracle:
    """
    Abstract class for the Oracle the Active Learner interacts with. 
    """

    # statistics on the iteraction with the oracle
    number_of_instances_labeled = None  # keep track of how many times the oracle was asked to label a meta-path

    def __init__(self):
        self.number_of_instances_labeled = 0

    # TODO Think about whether exclusion is own class or 0
    def rate_meta_path(self, meta_path: MetaPath) -> float:
        self.number_of_instances_labeled += 1
        return self._rate_meta_path(meta_path)

    def _rate_meta_path(self, meta_path: MetaPath) -> float:
        raise NotImplementedError("Should have implemented this.")

    def wants_to_continue(self) -> bool:
        raise NotImplementedError("Should have implemented this.")


class CmdLineOracle(Oracle):
    """
    CmdLineOracle interacts with the command line.
    """

    def _rate_meta_path(self, meta_path: MetaPath) -> float:
        print("Please rate this meta-path: {}".format(meta_path))
        rating = input()
        return float(rating)

    def wants_to_continue(self) -> bool:
        print("Do you want to continue rating? [y/n]")
        keep_going = input()
        if keep_going in 'no':
            return False
        return True


class MockOracle(Oracle):
    """
    MockOracle rates all meta-paths with 1 and never wants to continue to rate more.
    """

    def _rate_meta_path(self, meta_path: MetaPath) -> float:
        return 1.0

    def wants_to_continue(self) -> bool:
        return self.number_of_instances_labeled < 5


class SpecialistOracle(Oracle):
    """
    A  SpecialistOracle rates all meta-paths according to its rating dictionary.
    """

    rating = None
    maximal_queries = None  # maximal number of queries tolerated by the oracle

    def __init__(self, rating: Dict[MetaPath, float], maximal_queries: int):
        super.__init__(self)
        self.rating = rating
        self.maximal_queries = maximal_queries

    def _rate_meta_path(self, meta_path: MetaPath) -> float:
        return self.rating[meta_path]

    def wants_to_continue(self) -> bool:
        return self.number_of_instances_labeled < self.maximal_queries
