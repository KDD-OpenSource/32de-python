from active_learning.oracles import UserOracle, FunctionalOracle
from active_learning.evaluation import Evaluator
from active_learning.active_learner import RandomSelectionAlgorithm, GPSelect_Algorithm
from active_learning.rating import length_based
import unittest


class ActiveLearningExperimentsTest(unittest.TestCase):
    def test_UserOracle(self):
        """
        An example for using the UserOracle with a file generated from the OpenKDD day on the Rotten Tomato dataset.
        """
        print("Hear, hear!"
              "The Oracle of Merlin and Felix ... ")

        merlin_eval = Evaluator(dataset_name='Rotten Tomato',
                                oracle=UserOracle,
                                oracle_params={
                                    'ground_truth_path': 'rated_datasets/Rotten Tomato_Merlin_1519148528.2417703.json',
                                    'is_zero_indexed': False},
                                batch_size=5)
        felix_eval = Evaluator(dataset_name='Rotten Tomato',
                               oracle=UserOracle,
                               oracle_params={
                                   'ground_truth_path': 'rated_datasets/Rotten Tomato_mcfelix_1519142949.904623.json',
                                   'is_zero_indexed': False,
                                   'default_rating': 0.5},
                               batch_size=5)
        stats = merlin_eval.compute()
        stats = felix_eval.compute()
        print(stats)

    def test_randomSelection(self):
        """
        An example for using the MockOracle on the Rotten Tomato dataset.
        """
        eval = Evaluator(dataset_name='Rotten Tomato', algorithm=RandomSelectionAlgorithm, algo_params={},batch_size=5)
        stats = eval.compute()


    def test_gp_select(self):
        eval = Evaluator(dataset_name='Rotten Tomato',
                         oracle=FunctionalOracle,
                         oracle_params={'rating_func': length_based},
                         algorithm=GPSelect_Algorithm,
                         algo_params={'hypothesis': 'Gaussian Process', 'beta': 0.12})
        eval.compute()


if __name__ == '__main__':
    unittest.main()
