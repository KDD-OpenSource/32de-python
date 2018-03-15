from util.datastructures import MetaPath
from active_learning.active_learner import UncertaintySamplingAlgorithm, ActiveLearningAlgorithm

from typing import Dict, List, Callable
from abc import ABC, abstractmethod
from active_learning.rating import constant
import json

class Oracle(ABC):
    """
    Abstract class for the Oracle the Active Learner interacts with. 
    """

    def __init__(self):
        self.rating = {}

    def _rate_meta_paths(self, metapaths: List[Dict]) -> List[Dict]:
        """
        Rate a list of meta-paths according to their id.
        """
        return [{'id': mp['id'], 'rating': self._rate_meta_path(mp)} for mp in metapaths]

    @abstractmethod
    def _rate_meta_path(self, mp: Dict) -> float:
        """
        Rate a meta-path according to its importance to the oracle.
        """
        pass

    def _wants_to_continue(self) -> bool:
        """
        Determine whether the oracle wants to continue rating more paths.
        """
        return True


class FlexibleOracle(Oracle):
    """
    Flexible Oracle that rates all meta-paths according to .
    """

    def __init__(self, rating_func: Callable[[MetaPath], float]):
        # Set configuration of this oracle
        self.rating_func = rating_func
        super(FlexibleOracle, self).__init__()

    def _rate_meta_path(self, metapath: Dict) -> float:
        if metapath['id'] in self.rating.keys():
            return self.rating[metapath['id']]
        rating = self.rating_func(metapath['metapath'])
        self.rating[metapath['id']] = rating
        return rating


class UserOracle(Oracle):
    """
    An Oracle designed to use a json-file containing rated Meta-Paths as labels.
    """

    def __init__(self, ground_truth_path: str, default_rating=0.5, is_zero_indexed=True):
        # Set configuration of this oracle
        self.is_zero_indexed = is_zero_indexed
        self.default_rating = default_rating

        # Load the rating into the oracle
        self.rating = self.load_rating_from(ground_truth_path)
        super(UserOracle, self).__init__()

    def load_rating_from(self, ground_truth_path: str):
        """
        Loads a dataset of saved ratings.
        """
        data = json.load(open(ground_truth_path, "r", encoding="utf8"))
        rating = {}
        i = 0
        first = True
        for probably_path in data["meta_paths"]:
            # Ignore first time_to_rate
            if first:
                first = False
                continue
            i += 1
            if i == 6:
                # Ignore time_to_rate
                i = 0
            else:
                if 'time_to_rate' not in probably_path.keys():
                    rating[probably_path['id']] = probably_path['rating']
        return rating

    def _rate_meta_path(self, metapath: Dict) -> float:
        id = metapath['id'] if self.is_zero_indexed else metapath['id'] + 1
        try:
            return float(self.rating[id])
        except KeyError:
            return self.default_rating
