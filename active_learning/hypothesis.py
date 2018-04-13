from util.datastructures import MetaPath
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import numpy as np
import logging


class MPLengthHypothesis:
    """
    A Hypothesis over a rating of meta-paths. It decides which meta-path will be sent to the oracle next.

    This is just a simple example with a hypothesis, that the rating depends on the length of the meta-path and has a
    cutoff at some length. 
    """

    # If the rating for a meta path is above this threshold, the meta path length will be regarded as important.
    RATING_THRESHOLD = 0.3

    def __init__(self, maximum_interesting_length=4):
        self.maximum_interesting_length = maximum_interesting_length

    def update(self, meta_path: MetaPath, rating: float) -> None:
        """
        Update the hypothesis based on a new incoming rating.
        """
        if rating > self.RATING_THRESHOLD and len(meta_path) > self.maximum_interesting_length:
            self.maximum_interesting_length = len(meta_path)

    def predict_rating(self, meta_path: MetaPath) -> float:
        """
        :return: An estimated rating based on the current hypothesis.
        """
        if len(meta_path) > self.maximum_interesting_length:
            return 1.0
        else:
            return 0.0


class GaussianProcessHypothesis:
    def __init__(self, meta_paths, **hypothesis_params):
        self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        self.gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        if not 'embedding_strategy' in hypothesis_params:
            self.meta_paths = self._tfidf_transform(meta_paths)
        else:
            self.meta_paths = hypothesis_params['embedding_strategy'](meta_paths)
        self.similarity = kernel(self.meta_paths,self.meta_paths)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state

    def __setstate__(self, d):
        # Restore instance attributes
        self.__dict__.update(d)
        self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))

    def _length_based_transform(self, meta_paths):
        """
        Trivial transformation into feature space of length x unique_length.
        """
        return np.array([[len(mp), len(set(mp.as_list()))] for mp in meta_paths])

    def _tfidf_transform(self, meta_paths):
        """
        Transform the meta paths as tfidf vectors.
        """
        vectorizer = TfidfVectorizer(analyzer='word', token_pattern='\\b\\w+\\b')
        vectorizer.fit([str(mp) for mp in meta_paths])
        return vectorizer.transform(map(str, meta_paths)).toarray()

    def update(self, idx, ratings):
        if len(idx) == 0:
            return []
        self.gp.fit(self.meta_paths[idx], ratings)

    def predict_rating(self, idx):
        prediction = self.gp.predict(self.meta_paths[idx])
        self.logger.debug("prediction for {} is {}", self.meta_paths[idx], prediction)
        return self.gp.predict(self.meta_paths[idx])

    def get_uncertainty(self, idx):
        uncertainty_all_meta_paths = self.gp.predict(self.meta_paths[idx], return_std=True)[1]
        self.logger.debug("The uncertainty for the meta paths is: {}".format(uncertainty_all_meta_paths))
        return uncertainty_all_meta_paths

    def get_similarity(self):
        """
        Computes the complete similarity matrix according to the kernel.
        """
        return self.similarity
