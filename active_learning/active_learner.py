from .hypothesis import Hypothesis
from .meta_path_selector import RandomMetaPathSelector
from .oracles import MockOracle
from util.datastructures import MetaPathRatingGraph


# algorithm types
NO_USER_FEEDBACK = 'no_user_feedback'
ITERATIVE_BATCHING = 'iterative_batching'
COLLECT_ALL = 'collect_all'


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
        self.meta_path_selector = RandomMetaPathSelector(self.meta_paths, self.hypothesis)

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
