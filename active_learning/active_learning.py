from active_learning.oracles import MockOracle
from util.datastructures import MetaPath, MetaPathRatingGraph
from typing import List
import numpy as np

# algorithm types
NO_USER_FEEDBACK = 'no_user_feedback'
ITERATIVE_BATCHING = 'iterative_batching'
COLLECT_ALL = 'collect_all'


class Hypothesis:
    """
    A Hypothesis over a rating of meta-path.
    
    This is just a simple example with a hypothesis, that the rating depends on the length of the meta-path and has a 
    cutoff at some length.
    """

    # example: use length says how important the graph is
    exact_meta_path_rating = None
    params = None

    def __init__(self):
        self.params = {'max_interesting_length': 4}
        self.exact_meta_path_rating = {}

    def update(self, meta_path: MetaPath, rating: float) -> None:
        """
        Update the hypothesis based on a new incoming rating.
        """
        self.exact_meta_path_rating[meta_path] = rating
        if rating > 1 and len(meta_path) > self.params['max_interesting_length']:
            self.params['max_interesting_length'] = len(meta_path)

    def rank_meta_path(self, meta_path: MetaPath) -> float:
        """
        :return: An estimated rating based on the curent hypothesis. 
        """
        if len(meta_path) > self.params['max_interesting_length']:
            return 1.0
        else:
            return 0.0



class MetaPathSelector:
    """
    A MetaPathSelector selects the next meta-path to be rated.
    """

    VISITED = 0
    NOT_VISITED = 1

    hypothesis = None

    meta_paths = None
    visited = None

    def __init__(self, meta_paths: List[MetaPath], hypothesis: Hypothesis):
        self.meta_paths = np.array(meta_paths)
        self.visited = np.array([self.NOT_VISITED] * len(meta_paths))
        self.hypothesis = hypothesis

    def get_next(self, size=1) -> List[MetaPath]:
        """
        :return: requested number of next meta-paths to be shown.
        """
        idx = np.random.choice(range(len(self.meta_paths)), replace=False, p=self._prob_choose_mp(), size=(size))
        self.visited[idx] = self.VISITED
        return self.meta_paths[idx]

    def get_all(self) -> List[MetaPath]:
        return self.meta_paths

    def get_all_unrated(self) -> List[MetaPath]:
        return [mp for i, mp in enumerate(self.meta_paths) if self.visited[i] == self.NOT_VISITED]

    def _prob_choose_mp(self) -> np.ndarray:
        return self.visited / sum(self.visited)


class ActiveLearner:
    """
    Active Learner retrieves a rating for meta-paths by interacting with an oracle.
    """

    INFERED_RATING = 'infered rating'
    USER_ONLY_RATING = 'user-only rating'

    meta_paths = None
    algorithm_type = None
    oracle = None
    batch_size = None
    meta_path_selector = None
    hypothesis = None  # the hypothesis generated over all meta-paths from the current learning process

    def __init__(self, data_set, oracle=MockOracle(), algorithm=NO_USER_FEEDBACK, batch_size=5):
        self.meta_paths = data_set
        self.oracle = oracle
        self.algorithm_type = algorithm
        self.batch_size = batch_size
        self.hypothesis = Hypothesis()
        self.meta_path_selector = MetaPathSelector(self.meta_paths, self.hypothesis)

    def retrieve_rating(self,scope=USER_ONLY_RATING):
        self._rate_paths()
        rating = self.hypothesis.exact_meta_path_rating
        if scope == self.USER_ONLY_RATING:
            # only user ratings are returned
            return MetaPathRatingGraph(rating=rating)
        if scope == self.INFERED_RATING:
            # also use the predictions from the hypothesis on all other meta-paths
            for mp in self.meta_path_selector.get_all_unrated():
                rating[mp] = self.hypothesis.rank_meta_path(mp)
            return MetaPathRatingGraph(rating=rating)
        raise NotImplementedError("The scope {} is not defined.".format(scope))

    def _rate_paths(self):
        """
        Execute the specified algorithm on the current rating to retrieve a user rating.  
        """
        # consult oracle
        if self.algorithm_type == NO_USER_FEEDBACK:
            # do not let the user rate any meta_paths
            pass
        if self.algorithm_type == ITERATIVE_BATCHING:
            # produce rating of batches in random order, until user wants to stop
            while self.oracle.wants_to_continue():
                for mp in self.meta_path_selector.get_next(self.batch_size):
                    rating = self.oracle.rate_meta_path(meta_path=mp)
                    self.hypothesis.update(meta_path=mp, rating=rating)

        if self.algorithm_type == COLLECT_ALL:
            # disregard wishes of oracle and ask until every rating for each meta-path is known
            for mp in self.meta_path_selector.get_all():
                rating = self.oracle.rate_meta_path(meta_path=mp)
                self.hypothesis.update(meta_path=mp, rating=rating)
