from util.datastructures import MetaPath

# mock data
mock_paths = [MetaPath([1, 2], [3]), MetaPath([1, 2, 1], [3, 4]), MetaPath([1, 3, 2], [1, 1]),
              MetaPath([1, 3, 2], [1, 1]), MetaPath([1, 3, 2], [1, 1])]

# algorithm types
NO_RATING = 'no_rating'
BASELINE = 'baseline'


class Oracle:
    """
    Abstract class for the Oracle the Active Learner interacts with. 
    """

    def rate_meta_path(self, meta_path: MetaPath) -> float:
        raise NotImplementedError("Should have implemented this.")

    def wants_to_continue(self) -> bool:
        raise NotImplementedError("Sould have implemented this.")


class MockOracle(Oracle):
    """
    MockOracle rates all meta-paths with 1 and never wants to continue to rate more
    """

    def rate_meta_path(self, meta_path):
        return 1

    def wants_to_continue(self):
        return False


class ActiveLearner:
    meta_paths = None
    algorithm = None
    oracle = None
    batch_size = None

    def __init__(self, oracle=MockOracle(), algorithm=NO_RATING, batch_size=5):
        self.meta_paths = self.fetch_meta_paths()
        self.oracle = oracle
        self.algorithm = algorithm
        self.batch_size = batch_size

    def retrieve_user_rating(self):
        if self.algorithm == NO_RATING:
            # do let the user rate any meta_paths
            return []
        if self.algorithm == BASELINE:
            # take first 5
            interesting_meta_paths = self.meta_paths[:self.batch_size]
            ratings = map(self.oracle.rate_meta_path, interesting_meta_paths)
            return zip(ratings, interesting_meta_paths)

    def fetch_meta_paths(self):
        # TODO: use java API via cypher
        return mock_paths
