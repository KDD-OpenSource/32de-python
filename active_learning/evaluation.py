from active_learning.rating import constant
from active_learning.active_learner import UncertaintySamplingAlgorithm
from active_learning.oracles import FunctionalOracle
from util.meta_path_loader_dispatcher import MetaPathLoaderDispatcher

import logging
import pandas as pd

# Set up logging
logger = logging.getLogger()
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)  # logger.setLevel(logging.DEBUG)


class Evaluator:
    def __init__(self, dataset_name: str,
                 batch_size: int = 5,
                 algorithm=UncertaintySamplingAlgorithm,
                 algo_params={'hypothesis': 'Gaussian Process'},
                 oracle=FunctionalOracle,
                 oracle_params={'rating_func': constant},
                 seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed

        meta_path_loader = MetaPathLoaderDispatcher().get_loader(dataset_name)
        meta_paths = meta_path_loader.load_meta_paths()

        self.algorithm = algorithm(meta_paths=meta_paths, seed=seed, **algo_params)
        self.oracle = oracle(**oracle_params)

    def compute(self) -> pd.DataFrame:
        """
        Label the datapoints according to the ActiveLearningAlgorithm and collect statistics.
        Communication is via ids of meta-paths which are assumed to be zero-indexed.

        :return: pd.Dataframe with statistics about the collection process.
        """
        statistics = []
        is_last_batch = False

        while self.oracle._wants_to_continue() == True and not is_last_batch:
            # Retrieve next batch
            next_metapaths, is_last_batch = self.algorithm.get_next(batch_size=self.batch_size)
            ids_to_be_rated = [mp['id'] for mp in next_metapaths]
            logger.info("\tRating paths:\t{}".format(ids_to_be_rated))

            # Rate paths and update algorithm
            rated_metapaths = self.oracle._rate_meta_paths(next_metapaths)
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
        squared_values = [pow(self.oracle._rate_meta_path(p) - p['rating'], 2) for p in predictions]
        return sum(squared_values) / len(squared_values)