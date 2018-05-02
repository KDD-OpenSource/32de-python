from active_learning.oracles import UserOracle, FunctionalOracle
from active_learning.evaluation import Evaluator
from active_learning.active_learner import RandomSelectionAlgorithm, GPSelect_Algorithm, UncertaintySamplingAlgorithm
from active_learning.rating import length_based
import unittest


class ActiveLearningExperimentsTest(unittest.TestCase):

    def test_evaluator(self):
        algorithm = UncertaintySamplingAlgorithm
        algo_params_length = {'hypothesis': 'Gaussian Process', 'hypothesis_params': {'transformation': 'length'}}
        algo_params_tfidf = {'hypothesis': 'Gaussian Process', 'hypothesis_params': {'transformation': 'tfidf'}}

        logs_path = '/home/freya/BP/32de-python/notebooks/active_learning/logs/testrun/f'

        def rating_func_constant(c):
            return lambda x: c

        rating_func = rating_func_constant(1)
        oracle = FunctionalOracle(**{'rating_func': rating_func})

        res = Evaluator(algorithm=algorithm, algo_params=algo_params_tfidf,
                  oracle=oracle,
                  batch_size=1, dataset_name='Rotten Tomato', logs_path=logs_path).compute()


if __name__ == '__main__':
    unittest.main()
