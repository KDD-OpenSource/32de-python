from active_learning.oracles import UserOracle, FunctionalOracle
from active_learning.evaluation import Evaluator
from active_learning.active_learner import RandomSelectionAlgorithm, GPSelect_Algorithm, UncertaintySamplingAlgorithm
from active_learning.rating import length_based
import unittest


class ActiveLearningExperimentsTest(unittest.TestCase):
    def test_UserOracle(self):
        """
        An example for using the UserOracle with a file generated from the OpenKDD day on the Rotten Tomato dataset.
        """
        print("Hear, hear!"
              "The Oracle of Merlin and Felix ... ")

        merlin_eval = Evaluator('Rotten Tomato', 5, UncertaintySamplingAlgorithm,
                                UserOracle('rated_datasets/Rotten Tomato_Merlin_1519148528.2417703.json'))
        felix_eval = Evaluator('Rotten Tomato', 5, UncertaintySamplingAlgorithm,
                               UserOracle('rated_datasets/Rotten Tomato_mcfelix_1519142949.904623.json'))
        stats = merlin_eval.compute()
        stats = felix_eval.compute()
        print(stats)

    def test_randomSelection(self):
        """
        An example for using the MockOracle on the Rotten Tomato dataset.
        """
        eval = Evaluator('Rotten Tomato', 5, RandomSelectionAlgorithm, FunctionalOracle())
        stats = eval.compute()


    def test_gp_select(self):
        eval = Evaluator('Rotten Tomato', 5, GPSelect_Algorithm, FunctionalOracle(rating_func= length_based), beta=0.12)
        eval.compute()


if __name__ == '__main__':
    unittest.main()
