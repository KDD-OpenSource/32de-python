from util.datastructures import MetaPath
from active_learning.active_learner import UncertaintySamplingAlgorithm, ActiveLearningAlgorithm
from util.meta_path_loader_dispatcher import MetaPathLoaderDispatcher
from typing import Dict, List, Callable
from abc import ABC, abstractmethod
import json
import logging
import pandas as pd

# Set up logging
logger = logging.getLogger()
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)


# logger.setLevel(logging.DEBUG)


class Oracle(ABC):
    """
    Abstract class for the Oracle the Active Learner interacts with. 
    """

    def __init__(self, dataset_name: str, batch_size: int,
                 algorithm,
                 algo_params: Dict,
                 seed: int):
        self.batch_size = batch_size
        self.seed = seed

        meta_path_loader = MetaPathLoaderDispatcher().get_loader(dataset_name)
        meta_paths = meta_path_loader.load_meta_paths()

        self.algorithm = algorithm(meta_paths=meta_paths, seed=seed, **algo_params)

    def compute(self) -> pd.DataFrame:
        """
        Label the datapoints according to the ActiveLearningAlgorithm and collect statistics.
        Communication is via ids of meta-paths which are assumed to be zero-indexed.
        
        :return: pd.Dataframe with statistics about the collection process.
        """
        statistics = []
        is_last_batch = False

        while self._wants_to_continue() == True and not is_last_batch:
            # Retrieve next batch
            next_metapaths, is_last_batch = self.algorithm.get_next(batch_size=self.batch_size)
            ids_to_be_rated = [mp['id'] for mp in next_metapaths]
            logger.info("\tRating paths:\t{}".format(ids_to_be_rated))

            # Rate paths and update algorithm
            rated_metapaths = self._rate_meta_paths(next_metapaths)
            self.algorithm.update(rated_metapaths)

            # Log statistics
            mse = self.compute_mse()
            stats = {'mse': mse}
            statistics.append(stats)

            logger.info('\n'.join(["\t{}:\t{}".format(key, value) for key, value in stats.items()]))
            logger.info("")
        logger.info("Finished rating paths.")
        return pd.DataFrame.from_records(statistics)

    def compute_mse(self) -> float:
        """
        Calculate the Mean Squared Error between the predicted ratings and the ground truth.
        """
        predictions = self.algorithm.get_all_predictions()
        squared_values = [pow(self._rate_meta_path(p) - p['rating'], 2) for p in predictions]
        return sum(squared_values) / len(squared_values)

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
        raise NotImplementedError("Should have implemented this.")

    @abstractmethod
    def _wants_to_continue(self) -> bool:
        """
        Determine whether the oracle wants to continue rating more paths.
        """
        raise NotImplementedError("Should have implemented this.")


class CmdLineOracle(Oracle):
    """
    CmdLineOracle interacts with the command line.
    
    WARNING: This is still buggy, because you have to rate all the paths in one go regardless of batch size.
             No interactivity is given.
             Keeping it for development without user interface.
    
    """

    def __init__(self, dataset_name, batch_size, seed=42, algorithm=UncertaintySamplingAlgorithm,
                 algo_params={'hypothesis': 'Gaussian Process'}):
        self.rating = {}
        super(CmdLineOracle, self).__init__(dataset_name, batch_size, algorithm, algo_params, seed)

    def _rate_meta_path(self, metapath: Dict) -> float:
        if metapath['id'] in self.rating.keys():
            return self.rating[metapath['id']]
        print("Please rate this meta-path: {}".format(metapath['metapath']))
        rating = input()
        self.rating[metapath['id']] = rating
        return float(rating)

    def _wants_to_continue(self) -> bool:
        print("Do you want to continue rating? [y/n]")
        keep_going = input()
        if keep_going in 'no':
            return False
        return True


class MockOracle(Oracle):
    """
    MockOracle rates all meta-paths with 1.
    """

    def __init__(self, dataset_name: str, batch_size=5, seed=42, algorithm=UncertaintySamplingAlgorithm,
                 algo_params={'hypothesis': 'Gaussian Process'}):
        # Set configuration of this oracle
        self.batch_size = batch_size

        # Load the dataset and the algorithm to operate on
        meta_path_loader = MetaPathLoaderDispatcher().get_loader(dataset_name)
        meta_paths = meta_path_loader.load_meta_paths()
        self.algorithm = algorithm(meta_paths=meta_paths, seed=seed, **algo_params)

    def _rate_meta_path(self, meta_path: MetaPath) -> float:
        return 1.0

    def _wants_to_continue(self) -> bool:
        return True


class FlexibleOracle(Oracle):
    """
    Flexible Oracle that rates all meta-paths according to .
    """

    def __init__(self, dataset_name: str, rating_func: Callable[[MetaPath], float], batch_size=5, seed=42,
                 algorithm=UncertaintySamplingAlgorithm, algo_params={'hypothesis': 'Gaussian Process'}):
        # Set configuration of this oracle
        self.rating_func = rating_func
        self.rating = {}

        super(FlexibleOracle, self).__init__(dataset_name, batch_size, algorithm, algo_params, seed)

    def _rate_meta_path(self, metapath: Dict) -> float:
        if metapath['id'] in self.rating.keys():
            return self.rating[metapath['id']]
        rating = self.rating_func(metapath['metapath'])
        self.rating[metapath['id']] = rating
        return rating

    def _wants_to_continue(self) -> bool:
        return True


class UserOracle(Oracle):
    """
    An Oracle designed to use a json-file containing rated Meta-Paths as labels.
    """

    def __init__(self, dataset_name: str, ground_truth_path: str, batch_size: int = 5,
                 algorithm=UncertaintySamplingAlgorithm,
                 algo_params: Dict = {'hypothesis': 'Gaussian Process'},
                 seed: int = 42, default_rating=0.5,
                 is_zero_indexed=True):
        # Set configuration of this oracle
        self.is_zero_indexed = is_zero_indexed
        self.default_rating = default_rating

        # Load the rating into the oracle
        self.rating = self.load_rating_from(ground_truth_path)
        super(UserOracle, self).__init__(dataset_name, batch_size, algorithm, algo_params, seed)

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

    def _wants_to_continue(self) -> bool:
        return True
