from active_learning.rating import entropy
from util.datastructures import MetaPath
import unittest


class RatingsTest(unittest.TestCase):
    def test_UserOracle(self):
        mp = MetaPath(edge_node_list=['b', 'b', 'b'])
        rating = entropy(mp)
        self.assertEqual(rating, 0.0)

if __name__ == '__main__':
    unittest.main()
