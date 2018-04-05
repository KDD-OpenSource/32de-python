from abc import ABC, abstractmethod, ABCMeta
from enum import Enum
from typing import List

import numpy as np

from util.datastructures import MetaPath
from .hypothesis import GaussianProcessHypothesis

# algorithm types
NO_USER_FEEDBACK = 'no_user_feedback'
ITERATIVE_BATCHING = 'iterative_batching'
COLLECT_ALL = 'collect_all'


class State(Enum):
    VISITED = 0
    NOT_VISITED = 1


class AbstractActiveLearningAlgorithm(ABC):
    standard_rating = 0.5

    def __init__(self, meta_paths: List[MetaPath], seed: int):
        self.meta_paths = np.array(meta_paths)
        self.meta_paths_rating = np.array(np.zeros(len(meta_paths)))
        self.visited = np.array([State.NOT_VISITED] * len(meta_paths))
        self.random = np.random.RandomState(seed=seed)

    def has_one_batch_left(self, batch_size):
        if len(np.where(self.visited == State.NOT_VISITED)[0]) >= batch_size:
            return False
        return True

    def update(self, meta_paths):
        idx = [mp['id'] for mp in meta_paths]
        ratings = [mp['rating'] for mp in meta_paths]
        self.visited[idx] = State.VISITED
        self.meta_paths_rating[idx] = ratings

    def get_next(self, batch_size=1) -> (List[MetaPath], bool):
        """
        :return: requested number of next meta-paths to be rated next.
        """
        is_last_batch = self.has_one_batch_left(batch_size)
        if is_last_batch:
            batch_size = len(np.where(self.visited == State.NOT_VISITED)[0])
        ids = self._select(batch_size)

        mps = [{'id': int(meta_id),
                'metapath': meta_path,
                'rating': self.standard_rating} for meta_id, meta_path in
               zip(ids, self.meta_paths[ids])]

        return mps, is_last_batch

    @abstractmethod
    def _select(self, n):
        """
        Select the next n paths to be labeled
        """
        pass

    def create_output(self):
        mps = [{'id': int(meta_id[0]),
                'metapath': meta_path.as_list(),
                'rating': self.meta_paths_rating[meta_id]} for meta_id, meta_path in np.ndenumerate(self.meta_paths) if
               self.visited[meta_id] == State.VISITED]
        return mps


class RandomSelectionAlgorithm(AbstractActiveLearningAlgorithm):
    """
    An active learning algorithm, that asks for randomly labeled instances.
    """

    def __init__(self, meta_paths: List[MetaPath], standard_rating: float = 0.5, seed=42):
        self.standard_rating = standard_rating
        super(RandomSelectionAlgorithm, self).__init__(meta_paths, seed)

    def available_hypotheses(self):
        return [None]

    def _select(self, n):
        """
        Choose the next paths to be rated randomly.
        """
        return self.random.choice(range(len(self.meta_paths)), replace=False, p=self._prob_choose_meta_path(),
                                  size=n)

    def _predict(self, id):
        if self.visited[id] == State.VISITED:
            return self.meta_paths_rating[id]
        return sum(self.meta_paths_rating[np.where(self.visited == State.VISITED)]) / len(self.meta_paths_rating)

    def _predict_rating(self, idx):
        return [self._predict(id) for id in idx]

    def get_all_predictions(self):
        idx = range(len(self.meta_paths))
        return [{'id': id, 'rating': rating, 'metapath': self.meta_paths[id]} for id, rating in
                zip(idx, self._predict_rating(idx))]

    """
    Functions
    """

    def _prob_choose_meta_path(self) -> np.ndarray:
        not_visited = np.where(self.visited == State.NOT_VISITED)[0]
        if len(not_visited) > 0:
            probs = np.zeros(len(self.meta_paths))
            probs[not_visited] = 1.0
            return probs / len(not_visited)
        else:
            return np.append(np.zeros(len(self.meta_paths) - 1), [1])


class HypothesisBasedAlgorithm(AbstractActiveLearningAlgorithm, metaclass=ABCMeta):
    available_hypotheses = {'Gaussian Process': GaussianProcessHypothesis}

    def __init__(self, meta_paths: List[MetaPath], hypothesis: str, seed: int = 42, hypothesis_params = {}):
        self.hypothesis = GaussianProcessHypothesis(meta_paths=meta_paths,**hypothesis_params)
        super(HypothesisBasedAlgorithm, self).__init__(meta_paths, seed)

    def update(self, meta_paths):
        super(HypothesisBasedAlgorithm, self).update(meta_paths)
        self.hypothesis.update(np.where(self.visited == State.VISITED)[0],
                               self.meta_paths_rating[np.where(self.visited == State.VISITED)[0]])

    def _select(self, batch_size):
        criterion = self.compute_selection_criterion()
        criterion = criterion[np.where(self.visited == State.NOT_VISITED)]
        most_uncertain_idx = np.argpartition(criterion, -batch_size)[-batch_size:]
        most_uncertain_ids = np.where(self.visited == State.NOT_VISITED)[0][most_uncertain_idx]
        return most_uncertain_ids

    @abstractmethod
    def compute_selection_criterion(self):
        """
        Select the next metapaths based on the uncertainty of them in the current model.
        """
        pass

    def get_all_predictions(self):
        idx = range(len(self.meta_paths))
        return [{'id': id, 'rating': rating, 'metapath': self.meta_paths[id]} for id, rating in
                zip(idx, self.hypothesis.predict_rating(idx))]


class UncertaintySamplingAlgorithm(HypothesisBasedAlgorithm):
    """
        An active learning algorithm, that requests labels on the data he is most uncertain of.
    """

    @staticmethod
    def options():
        return {}

    def compute_selection_criterion(self):
        """
        Select the next metapaths based on the uncertainty of them in the current model.
        """
        std = self.hypothesis.predict_std(range(len(self.meta_paths)))
        return std

class RandomSamplingAlgorithm(HypothesisBasedAlgorithm):
    """
        An active learning algorithm, that requests labels on the data he is most uncertain of.
    """

    @staticmethod
    def options():
        return {}

    def compute_selection_criterion(self):
        """
        Select the next metapaths based on the uncertainty of them in the current model.
        """
        random_criterion = self.random.randn(len(self.meta_paths))
        return random_criterion

class GPSelect_Algorithm(HypothesisBasedAlgorithm):
    """
        An active learning algorithm, that selects the most uncertain data, but prefers more highly rated datapoints.
        The parameter beta weights the trade-off between exploitation and exploration.
        ...
        Hastagiri P. Vanchinathan, Andreas Marfurt, Charles-Antoine Robelin, Donald Kossmann, and Andreas Krause. 2015. 
        Discovering Valuable items from Massive Data. In Proceedings of the 21th ACM SIGKDD (KDD '15).
        ACM, New York, NY, USA, 1195-1204. DOI: https://doi.org/10.1145/2783258.2783360
    """

    def __init__(self, meta_paths: List[MetaPath], hypothesis: str, beta: float = 0.5, seed: int = 42):
        super(GPSelect_Algorithm, self).__init__(meta_paths, hypothesis, seed)
        self.beta = beta

    @staticmethod
    def options():
        return {
            'hypothesis': [GaussianProcessHypothesis]
        }

    def compute_selection_criterion(self):
        """
        Select the next metapaths based on the uncertainty of them in the current model.
        """
        all_paths = range(len(self.meta_paths))
        criterion = np.sqrt(self.beta) * self.hypothesis.predict_std(all_paths) \
                    + self.hypothesis.predict_rating(all_paths)
        return criterion
