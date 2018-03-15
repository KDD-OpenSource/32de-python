from active_learning.oracles import UserOracle, MockOracle, FlexibleOracle
from active_learning.rating import length_based
import unittest


class ActiveLearningExperimentsTest(unittest.TestCase):
    def test_UserOracle(self):
        """
        An example for using the UserOracle with a file generated from the OpenKDD day on the Rotten Tomato dataset.
        """
        print("Hear, hear!"
              "The Oracle of Merlin and Felix ... ")

        merlin = UserOracle(dataset_name='Rotten Tomato',
                            ground_truth_path='rated_datasets/Rotten Tomato_Merlin_1519148528.2417703.json',
                            is_zero_indexed=False,
                            batch_size=5)
        felix = UserOracle(dataset_name='Rotten Tomato',
                           ground_truth_path='rated_datasets/Rotten Tomato_mcfelix_1519142949.904623.json',
                           is_zero_indexed=False,
                           batch_size=5,
                           default_rating=0.5)
        stats = merlin.compute()
        stats = felix.compute()

    def test_mockOracle(self):
        """
        An example for using the MockOracle on the Rotten Tomato dataset.
        """
        oracle = MockOracle(dataset_name='Rotten Tomato')
        stats = oracle.compute()
    def test_flexibleOracle(self):
        """
        Using the a FlexibleOracle for rating the Rotten Tomato dataset.
        """

        oracle = FlexibleOracle(dataset_name='Rotten Tomato', rating_func=length_based)
        stats = oracle.compute()
        print(stats)

        oracle = FlexibleOracle(dataset_name='Rotten Tomato', rating_func=length_based)
        stats = oracle.compute()
        print(stats)


if __name__ == '__main__':
    unittest.main()
