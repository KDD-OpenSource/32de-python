from util.datastructures import MetaPath
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import numpy as np


class Hypothesis:
    """
    A Hypothesis over a rating of meta-paths. It decides which meta-path will be sent to the oracle next.

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


class GaussianProcessHypothesis:
    def __init__(self, meta_paths, transformation='length'):
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        self.gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        if transformation == 'length':
            self.meta_paths = self._length_based_transform(meta_paths)
        elif transformation == 'tfidf':
            self.meta_paths = self._tfidf_transform(meta_paths)
        else:
            raise ValueError("'{}' is not a valid transformation.".format(transformation))

    def _length_based_transform(self, meta_paths):
        """
        Trivial transformation into feature space of length x unique_length.
        """
        return np.array([[len(mp), len(set(mp.as_list()))] for mp in meta_paths])

    def _tfidf_transform(self, meta_paths):
        """
        Transform the metapaths as tfidf vectors.
        """
        vectorizer = TfidfVectorizer(analyzer='word', token_pattern='\\b\\w+\\b')
        vectorizer.fit([str(mp) for mp in meta_paths])
        return vectorizer.transform(map(str, meta_paths)).toarray()

    @staticmethod
    def options():
        return {'transformation': ['length', 'tfidf']}

    def update(self, idx, ratings):
        if len(idx) == 0:
            return []
        self.gp.fit(self.meta_paths[idx], ratings)

    def predict_rating(self, idx):
        return self.gp.predict(self.meta_paths[idx])

    def predict_std(self, idx):
        return self.gp.predict(self.meta_paths[idx], return_std=True)[1]
