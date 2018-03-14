from .hypothesis import Hypothesis, GaussianProcessHypothesis
from abc import ABC, abstractmethod
from typing import List
from util.datastructures import MetaPath
import numpy as np

# algorithm types
NO_USER_FEEDBACK = 'no_user_feedback'
ITERATIVE_BATCHING = 'iterative_batching'
COLLECT_ALL = 'collect_all'


class ActiveLearningAlgorithm(ABC):
    @abstractmethod
    def __init__(self, data_set, model, interaction_config, seed_size):
        self.data_set = data_set
        self.model = model
        self.interaction_config = interaction_config
        return self.initial_label_request(seed_size)  # TODO better word for request

    @abstractmethod
    def initial_label_request(self, seed_size: int) -> List[MetaPath]:
        """
        Select the first meta-paths that should be labelled.
        """
        raise NotImplementedError("Subclass should implement.")

    @abstractmethod
    def available_hypotheses(self) -> List[Hypothesis]:
        """
        Return all hypothesises that are compatible with this ActiveLearningAlgorithm.
        """
        raise NotImplementedError("Subclass should implement.")

    @abstractmethod
    def update(self, meta_paths):
        """
        Receive new rated meta-paths and use them for criticising the hypothesis.
        """
        raise NotImplementedError("Subclass should implement.")

    @abstractmethod
    def get_next(self, batch_size: int) -> List[MetaPath]:
        """
        Select the next n meta-paths that should be rated.
        """
        raise NotImplementedError("Subclass should implement.")

    @abstractmethod
    def create_output(self) -> List[tuple]:
        """
        Make an output for whatever #TODO
        """
        raise NotImplementedError("Subclass should implement.")


class RandomSelectionAlgorithm(ActiveLearningAlgorithm):
    """
        An active learning algorithm, that asks for randomly labeled instances.
    """

    VISITED = 0
    NOT_VISITED = 1
    standard_rating = 0.5
    random = None

    def __init__(self, meta_paths: List[MetaPath], hypothesis: Hypothesis = None, seed=42):
        self.hypothesis = hypothesis
        assert hypothesis in self.available_hypotheses(), "Hypothesis {} not supported for this type of algorithm.".format(
            hypothesis)
        self.meta_paths = np.array(meta_paths)
        self.meta_paths_rating = np.array(np.zeros(len(meta_paths)))
        self.visited = np.array([self.NOT_VISITED] * len(meta_paths))
        self.random = np.random.RandomState(seed=seed)

    def available_hypotheses(self):
        return [None]

    def initial_label_request(self, seed_size):
        return self.get_next(batch_size=seed_size)

    def update(self, meta_paths):
        idx = [mp['id'] for mp in meta_paths]
        ratings = [mp['rating'] for mp in meta_paths]
        self.visited[idx] = self.VISITED
        self.meta_paths_rating[idx] = ratings

    def get_next(self, batch_size=1) -> (List[MetaPath], bool):
        """
        :return: requested number of next meta-paths to be shown.
        """
        is_last_batch = False
        if sum(self.visited) < batch_size:
            batch_size = sum(self.visited)
            is_last_batch = True

        idx = self.random.choice(range(len(self.meta_paths)), replace=False, p=self._prob_choose_meta_path(),
                                 size=batch_size)
        mps = [{'id': int(meta_id),
                'metapath': meta_path.as_list(),
                'rating': self.standard_rating} for meta_id, meta_path in zip(idx, self.meta_paths[idx])]
        return mps, is_last_batch

    def create_output(self):
        mps = [{'id': int(meta_id[0]),
                'metapath': meta_path.as_list(),
                'rating': self.meta_paths_rating[meta_id]} for meta_id, meta_path in np.ndenumerate(self.meta_paths) if
               self.visited[meta_id] == self.VISITED]
        return mps

    """
    Functions
    """

    def _prob_choose_meta_path(self) -> np.ndarray:

        if sum(self.visited):
            return self.visited / sum(self.visited)
        else:
            return np.append(np.zeros(len(self.meta_paths) - 1), [1])


class UncertaintySamplingAlgorithm(ActiveLearningAlgorithm):
    """
        An active learning algorithm, that requests labels on the data he is most uncertain of.
    """

    VISITED = 0
    NOT_VISITED = 1
    standard_rating = 0.5
    random = None
    available_hypotheses = {'Gaussian Process': GaussianProcessHypothesis}

    def __init__(self, meta_paths: List[MetaPath], hypothesis: str, seed=42):

        # TODO this might not work
        assert hypothesis in self.available_hypotheses.keys(), "Hypothesis {} not supported for this type of algorithm.".format(
            hypothesis)
        self.meta_paths = np.array(meta_paths)
        self.meta_paths_rating = np.array(np.zeros(len(meta_paths)))
        self.visited = np.array([self.NOT_VISITED] * len(meta_paths))
        self.random = np.random.RandomState(seed=seed)
        self.hypothesis = self.available_hypotheses[hypothesis](meta_paths=meta_paths)

    def initial_label_request(self, seed_size):
        return self.get_next(batch_size=seed_size)

    def update(self, meta_paths):
        idx = [mp['id'] for mp in meta_paths]
        ratings = [mp['rating'] for mp in meta_paths]
        self.visited[idx] = self.VISITED
        self.meta_paths_rating[idx] = ratings
        self.hypothesis.update(np.where(self.visited == self.VISITED), self.meta_paths_rating[np.where(self.visited == self.VISITED)])

    def has_one_batch_left(self, batch_size):
        if sum(self.visited) < batch_size:
            return True
        return False

    def get_next(self, batch_size=1) -> (List[MetaPath], bool):
        """
        :return: requested number of next meta-paths to be shown.
        """
        is_last_batch = self.has_one_batch_left(batch_size)

        if is_last_batch:
            batch_size = sum(self.visited)

        std = self.hypothesis.predict_std(range(len(self.meta_paths)))
        std = std[np.where(self.visited)]
        most_uncertain_idx = np.argpartition(std, -batch_size)[-batch_size:]
        most_uncertain_ids = np.where(self.visited)[0][most_uncertain_idx]
        mps = [{'id': int(meta_id),
                'metapath': meta_path,
                'rating': self.standard_rating} for meta_id, meta_path in
               zip(most_uncertain_ids, self.meta_paths[most_uncertain_idx])]
        return mps, is_last_batch

    def get_all_predictions(self):
        idx = range(len(self.meta_paths))
        return [{'id': id, 'rating': rating, 'metapath': self.meta_paths[id]} for id, rating in zip(idx, self.hypothesis.predict_rating(idx))]

    def create_output(self):
        mps = [{'id': int(meta_id[0]),
                'metapath': meta_path.as_list(),
                'rating': self.meta_paths_rating[meta_id]} for meta_id, meta_path in np.ndenumerate(self.meta_paths) if
               self.visited[meta_id] == self.VISITED]
        return mps
