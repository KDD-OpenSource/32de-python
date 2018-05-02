from active_learning.rating import *
from active_learning.active_learner import *
from active_learning.oracles import *
from util.meta_path_loader_dispatcher import MetaPathLoaderDispatcher

import logging
import pandas as pd
import util.tensor_logging as tf_log
import embeddings.meta2vec

# Set up logging
logger = logging.getLogger()
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)  # logger.setLevel(logging.DEBUG)

# Different parametrisations that can be used in experiments
ORACLES = [UserOracle, FunctionalOracle]
ALGORITHMS = [UncertaintySamplingAlgorithm, GPSelect_Algorithm, RandomSamplingAlgorithm]


class Evaluator:
    def __init__(self, dataset_name: str,
                 batch_size: int,
                 algorithm,
                 oracle,
                 seed: int = 42, **evaluator_params):
        # add tf logging
        self._tf_logger = tf_log.get_logger('evaluator')
        self._tf_logger.track_scalar('mse')
        self._tf_logger.track_scalar('abs')
        self._tf_logger.track_histogram('uncertainty')
        self._tf_logger.track_histogram('rating')
        self._tf_logger.start_writer()

        self.batch_size = batch_size

        meta_path_loader = MetaPathLoaderDispatcher().get_loader(dataset_name)
        meta_paths = meta_path_loader.load_meta_paths()

        # TODO find unique names
        # create mps
        mp_list = [MetaPath(edge_node_list=[hash(i) for i in mp.as_list()]) for mp in meta_paths]

        print('run metapath-embedding')
        embed = embeddings.meta2vec.calculate_metapath_embeddings(mp_list, metapath_embedding_size=10)
        [mp.store_embedding(embed[i][1]) for i, mp in enumerate(meta_paths)]
        print('end metapath-embedding')

        print(meta_paths)
        self.algorithm = algorithm(meta_paths=meta_paths, seed=seed, tf_logger=self._tf_logger,**evaluator_params)
        self.oracle = oracle

    def compute(self) -> pd.DataFrame:
        """
        Label the datapoints according to the ActiveLearningAlgorithm and collect statistics.
        Communication is via ids of meta-paths which are assumed to be zero-indexed.

        :return: pd.Dataframe with statistics about the collection process.
        """
        statistics = []
        is_last_batch = False
        while self.oracle._wants_to_continue() is True and not is_last_batch:
            # Retrieve next batch
            next_metapaths, is_last_batch, ref_paths = self.algorithm.get_next(batch_size=self.batch_size)
            ids_to_be_rated = [mp['id'] for mp in next_metapaths]
            logger.info("\tRating paths:\t{}".format(ids_to_be_rated))

            # Rate paths and update algorithm
            rated_metapaths = self.oracle._rate_meta_paths(next_metapaths)
            self.algorithm.update(rated_metapaths)

            # Log statistics
            mse = self.compute_mse()
            abs_diff = self.compute_absolute_error()
            stats = {'mse': mse,
                     'absolute_error': abs_diff}
            statistics.append(stats)
            self._tf_logger.update('mse', mse)
            self._tf_logger.update('abs', abs_diff)
            self._tf_logger.write_summary()

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

    def compute_absolute_error(self) -> float:
        """
        Calculate the Mean Squared Error between the predicted ratings and the ground truth.
        """
        predictions = self.algorithm.get_all_predictions()
        absolute_differences = [abs(self.oracle._rate_meta_path(p) - p['rating']) for p in predictions]
        return sum(absolute_differences) / len(absolute_differences)
