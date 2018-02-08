from util.datastructures import MetaPath

class Hypothesis:
    """
    A Hypothesis over a rating of meta-path. It decides which meta-path will be sent to the oracle next.

    This is just a simple example with a hypothesis, that the rating depends on the length of the meta-path and has a
    cutoff at some length. 
    """

    # example: use length says how important the graph is

    def __init__(self, maximum_interesting_length=4):
        self.params = {'max_interesting_length': maximum_interesting_length}
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
