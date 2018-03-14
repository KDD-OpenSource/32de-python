from active_learning.oracles import UserOracle, MockOracle
import matplotlib.pyplot as plt
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
                            batch_size=3)
        felix = UserOracle(dataset_name='Rotten Tomato',
                           ground_truth_path='rated_datasets/Rotten Tomato_mcfelix_1519142949.904623.json',
                           is_zero_indexed=False,
                           batch_size=3,
                           default_rating=0.5)
        stats = merlin.compute()
        #stats['mse'].plot(title='Merlins Rating')
        #plt.xlabel('iteration')
        #plt.ylabel('mean squared error')
        #plt.show()

        stats = felix.compute()
        #stats['mse'].plot(title='Felix Rating')
        #plt.xlabel('iteration')
        #plt.ylabel('mean squared error')
        #plt.show()

    def test_mockOracle(self):
        """
        An example for using the MockOracle on the Rotten Tomato dataset.
        """
        oracle = MockOracle(dataset_name='Rotten Tomato')
        stats = oracle.compute()
        #stats['mse'].plot(title='MSE')
        #plt.xlabel('iteration')
        #plt.ylabel('mean squared error')
        #plt.show()


if __name__ == '__main__':
    unittest.main()
