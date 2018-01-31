from active_learning.oracles import MockOracle
from util.datastructures import MetaPath, MetaPathRatingGraph

# algorithm types
NO_USER_FEEDBACK = 'no_user_feedback'
ITERATIVE_BATCHING = 'baseline'


class ActiveLearner:
    """
    Active Learner retrieves a rating for meta-paths by interacting with an oracle.
    """
    meta_paths = None
    algorithm_type = None
    oracle = None
    batch_size = None
    rating = None

    def __init__(self, oracle=MockOracle(), algorithm=NO_USER_FEEDBACK, batch_size=5, data_set=False):
        self.meta_paths = data_set if data_set else self.fetch_meta_paths()
        self.oracle = oracle
        self.algorithm_type = algorithm
        self.batch_size = batch_size
        self.rating = MetaPathRatingGraph()

    def retrieve_user_rating(self):
        self._rate_paths()
        self.rating.all_pair_distances()
        return self.rating

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
            interesting_meta_paths = self.meta_paths[:self.batch_size]

            while self.oracle.wants_to_continue():
                interesting_meta_paths = self.meta_paths[:self.batch_size]
                ratings = map(self.oracle.rate_meta_path, interesting_meta_paths)
                ordered = sorted(zip(ratings, interesting_meta_paths), key=lambda tuple: tuple[0])
                list(map(lambda rated_mp_1, rated_mp_2: self.rating.add_user_rating(rated_mp_1[1], rated_mp_2[1], rated_mp_2[0] - rated_mp_1[0]), ordered[:-1], ordered[1:]))

        # TODO: Transitive closure


    def fetch_meta_paths(self):
        # TODO: use java API via cypher
        mock_paths = [MetaPath([1, 2], [3]), MetaPath([1, 2, 1], [3, 4]), MetaPath([1, 3, 2], [1, 1]),
                      MetaPath([1, 3, 2], [1, 1]), MetaPath([1, 3, 2], [1, 1])]
        return mock_paths
