from util.datastructures import MetaPath, MetaPathRatingGraph

# algorithm types
NO_USER_FEEDBACK = 'no_user_feedback'
ITERATIVE_BATCHING = 'baseline'


class Oracle:
    """
    Abstract class for the Oracle the Active Learner interacts with. 
    """
    number_of_calls = None

    def __init__(self):
        self.number_of_calls = 0

    # TODO Think about whether exclusion is own class or 0
    def rate_meta_path(self, meta_path: MetaPath) -> float:
        self.number_of_calls += 1
        return self._rate_meta_path(meta_path)

    def _rate_meta_path(self, meta_path: MetaPath) -> float:
        raise NotImplementedError("Should have implemented this.")

    def wants_to_continue(self) -> bool:
        raise NotImplementedError("Sould have implemented this.")


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
        return self.number_of_calls < 5


class ActiveLearner:
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
                list(map(lambda x, y: self.rating.add_user_rating(x[1], y[1], y[0] - x[0]), ordered[:-1], ordered[1:]))

        # TODO: Transitive closure
        return self.rating


    def fetch_meta_paths(self):
        # TODO: use java API via cypher
        mock_paths = [MetaPath([1, 2], [3]), MetaPath([1, 2, 1], [3, 4]), MetaPath([1, 3, 2], [1, 1]),
                      MetaPath([1, 3, 2], [1, 1]), MetaPath([1, 3, 2], [1, 1])]
        return mock_paths
